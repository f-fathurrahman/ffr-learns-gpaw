from __future__ import annotations

from typing import Generator

import _gpaw
import numpy as np
from ase.dft.bandgap import bandgap
from ase.io.ulm import Writer
from ase.units import Bohr, Ha
from my_gpaw.gpu import synchronize
from my_gpaw.gpu.mpi import CuPyMPI
from my_gpaw.mpi import MPIComm, serial_comm
from my_gpaw.new import zip
from my_gpaw.new.brillouin import IBZ
from my_gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from my_gpaw.new.potential import Potential
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from my_gpaw.new.wave_functions import WaveFunctions
from my_gpaw.typing import Array1D, Array2D


def create_ibz_wave_functions(ibz: IBZ,
                              nelectrons: float,
                              ncomponents: int,
                              create_wfs_func,
                              kpt_comm: MPIComm = serial_comm
                              ) -> IBZWaveFunctions:
    """Collection of wave function objects for k-points in the IBZ."""
    rank_k = ibz.ranks(kpt_comm)
    mask_k = (rank_k == kpt_comm.rank)
    k_q = np.arange(len(ibz))[mask_k]

    nspins = ncomponents % 3

    wfs_qs: list[list[WaveFunctions]] = []
    for q, k in enumerate(k_q):
        wfs_s = []
        for spin in range(nspins):
            wfs = create_wfs_func(spin, q, k,
                                  ibz.kpt_kc[k], ibz.weight_k[k])
            wfs_s.append(wfs)
        wfs_qs.append(wfs_s)

    return IBZWaveFunctions(ibz,
                            nelectrons,
                            ncomponents,
                            wfs_qs,
                            kpt_comm)


class IBZWaveFunctions:
    def __init__(self,
                 ibz: IBZ,
                 nelectrons: float,
                 ncomponents: int,
                 wfs_qs: list[list[WaveFunctions]],
                 kpt_comm: MPIComm = serial_comm):
        """Collection of wave function objects for k-points in the IBZ."""
        self.ibz = ibz
        self.kpt_comm = kpt_comm
        self.nelectrons = nelectrons
        self.ncomponents = ncomponents
        self.collinear = (ncomponents != 4)
        self.spin_degeneracy = ncomponents % 2 + 1
        self.nspins = ncomponents % 3

        self.rank_k = ibz.ranks(kpt_comm)

        self.wfs_qs = wfs_qs

        self.q_k = {}  # IBZ-index to local index
        for wfs in self:
            self.q_k[wfs.k] = wfs.q

        self.band_comm = wfs.band_comm
        self.domain_comm = wfs.domain_comm
        self.dtype = wfs.dtype
        self.nbands = wfs.nbands

        self.fermi_levels: Array1D | None = None  # hartree

        self.energies: dict[str, float] = {}  # hartree

        self.xp = self.wfs_qs[0][0].xp
        if self.xp is not np:
            if not getattr(_gpaw, 'gpu_aware_mpi', False):
                self.kpt_comm = CuPyMPI(self.kpt_comm)

    def get_max_shape(self, global_shape: bool = False) -> tuple[int, ...]:
        """Find the largest wave function array shape.

        For a PW-calculation, this shape could depend on k-point.
        """
        if global_shape:
            shape = np.array(max(wfs.array_shape(global_shape=True)
                                 for wfs in self))
            self.kpt_comm.max(shape)
            return tuple(shape)
        return max(wfs.array_shape() for wfs in self)

    def is_master(self):
        return (self.domain_comm.rank == 0 and
                self.band_comm.rank == 0 and
                self.kpt_comm.rank == 0)

    def __str__(self):
        shape = self.get_max_shape(global_shape=True)
        wfs = self.wfs_qs[0][0]
        nbytes = (len(self.ibz) *
                  self.nbands *
                  len(self.wfs_qs[0]) *
                  wfs.bytes_per_band)
        ncores = (self.kpt_comm.size *
                  self.domain_comm.size *
                  self.band_comm.size)
        return (f'{self.ibz.symmetries}\n'
                f'{self.ibz}\n'
                f'{wfs._short_string(shape)}\n'
                f'spin-components: {self.ncomponents}'
                '  # (' +
                ('' if self.collinear else 'non-') + 'collinear spins)\n'
                f'bands: {self.nbands}\n'
                f'valence electrons: {self.nelectrons}\n'
                f'spin-degeneracy: {self.spin_degeneracy}\n'
                f'dtype: {self.dtype}\n\n'
                'memory:\n'
                f'    wave functions: {nbytes:_}  # bytes '
                f' ({nbytes // ncores:_} per core)\n\n'
                'parallelization:\n'
                f'    kpt:    {self.kpt_comm.size}\n'
                f'    domain: {self.domain_comm.size}\n'
                f'    band:   {self.band_comm.size}\n')

    def __iter__(self) -> Generator[WaveFunctions, None, None]:
        for wfs_s in self.wfs_qs:
            yield from wfs_s

    def move(self, fracpos_ac, atomdist):
        self.ibz.symmetries.check_positions(fracpos_ac)
        self.energies.clear()
        for wfs in self:
            wfs.move(fracpos_ac, atomdist)

    def orthonormalize(self, work_array_nX: np.ndarray = None):
        for wfs in self:
            wfs.orthonormalize(work_array_nX)

    def calculate_occs(self, occ_calc, fixed_fermi_level=False):
        degeneracy = self.spin_degeneracy

        # u index is q and s combined
        occ_un, fermi_levels, e_entropy = occ_calc.calculate(
            nelectrons=self.nelectrons / degeneracy,
            eigenvalues=[wfs.eig_n * Ha for wfs in self],
            weights=[wfs.weight for wfs in self],
            fermi_levels_guess=(None
                                if self.fermi_levels is None else
                                self.fermi_levels * Ha))

        if not fixed_fermi_level:
            self.fermi_levels = np.array(fermi_levels) / Ha
        else:
            assert self.fermi_levels is not None

        for occ_n, wfs in zip(occ_un, self):
            wfs._occ_n = occ_n

        e_entropy *= degeneracy / Ha
        e_band = 0.0
        for wfs in self:
            e_band += wfs.occ_n @ wfs.eig_n * wfs.weight * degeneracy
        e_band = self.kpt_comm.sum(float(e_band))  # XXX CPU float?

        self.energies = {
            'band': e_band,
            'entropy': e_entropy,
            'extrapolation': e_entropy * occ_calc.extrapolate_factor}

    def add_to_density(self, nt_sR, D_asii) -> None:
        """Compute density from wave functions and add to ``nt_sR``
        and ``D_asii``."""
        for wfs in self:
            wfs.add_to_density(nt_sR, D_asii)

        if self.xp is not np:
            synchronize()
        self.kpt_comm.sum(nt_sR.data)
        self.kpt_comm.sum(D_asii.data)

    def get_all_electron_wave_function(self,
                                       band,
                                       kpt=0,
                                       spin=0,
                                       grid_spacing=0.05,
                                       skip_paw_correction=False):
        wfs = self.get_wfs(kpt=kpt, spin=spin, n1=band, n2=band + 1)
        if wfs is None:
            return None
        assert isinstance(wfs, PWFDWaveFunctions)
        psit_X = wfs.psit_nX[0].to_pbc_grid()
        grid = psit_X.desc.uniform_grid_with_grid_spacing(grid_spacing)
        psi_r = psit_X.interpolate(grid=grid)

        if not skip_paw_correction:
            dphi_aj = wfs.setups.partial_wave_corrections()
            dphi_air = grid.atom_centered_functions(dphi_aj, wfs.fracpos_ac)
            dphi_air.add_to(psi_r, wfs.P_ani[:, 0])

        return psi_r

    def get_wfs(self,
                *,
                kpt: int = 0,
                spin: int = 0,
                n1=0,
                n2=0):
        rank = self.rank_k[kpt]
        if rank == self.kpt_comm.rank:
            wfs = self.wfs_qs[self.q_k[kpt]][spin]
            wfs2 = wfs.collect(n1, n2)
            if rank == 0:
                return wfs2
            if wfs2 is not None:
                wfs2.send(self.kpt_comm, 0)
            return
        master = (self.kpt_comm.rank == 0 and
                  self.domain_comm.rank == 0 and
                  self.band_comm.rank == 0)
        if master:
            return self.wfs_qs[0][0].receive(self.kpt_comm, rank)

    def get_eigs_and_occs(self, k=0, s=0):
        if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
            rank = self.rank_k[k]
            if rank == self.kpt_comm.rank:
                wfs = self.wfs_qs[self.q_k[k]][s]
                if rank == 0:
                    return wfs._eig_n, wfs._occ_n
                self.kpt_comm.send(wfs._eig_n, 0)
                self.kpt_comm.send(wfs._occ_n, 0)
            elif self.kpt_comm.rank == 0:
                eig_n = np.empty(self.nbands)
                occ_n = np.empty(self.nbands)
                self.kpt_comm.receive(eig_n, rank)
                self.kpt_comm.receive(occ_n, rank)
                return eig_n, occ_n
        return np.zeros(0), np.zeros(0)

    def get_all_eigs_and_occs(self):
        nkpts = len(self.ibz)
        if self.is_master():
            eig_skn = np.empty((self.nspins, nkpts, self.nbands))
            occ_skn = np.empty((self.nspins, nkpts, self.nbands))
        else:
            eig_skn = np.empty((0, 0, 0))
            occ_skn = np.empty((0, 0, 0))
        for k in range(nkpts):
            for s in range(self.nspins):
                eig_n, occ_n = self.get_eigs_and_occs(k, s)
                if self.is_master():
                    eig_skn[s, k, :] = eig_n
                    occ_skn[s, k, :] = occ_n
        return eig_skn, occ_skn

    def forces(self, potential: Potential) -> Array2D:
        F_av = self.xp.zeros((potential.dH_asii.natoms, 3))
        for wfs in self:
            wfs.force_contribution(potential, F_av)
        if self.xp is not np:
            synchronize()
        self.kpt_comm.sum(F_av)
        return F_av

    def write(self,
              writer: Writer,
              skip_wfs: bool) -> None:
        """Write fermi-level(s), eigenvalues, occupation numbers, ...

        ... k-points, symmetry information, projections and possibly
        also the wave functions.
        """
        eig_skn, occ_skn = self.get_all_eigs_and_occs()
        assert self.fermi_levels is not None
        writer.write(fermi_levels=self.fermi_levels * Ha,
                     eigenvalues=eig_skn * Ha,
                     occupations=occ_skn)
        ibz = self.ibz
        writer.child('kpts').write(
            atommap=ibz.symmetries.a_sa,
            bz2ibz=ibz.bz2ibz_K,
            bzkpts=ibz.bz.kpt_Kc,
            ibzkpts=ibz.kpt_kc,
            rotations=ibz.symmetries.rotation_svv,
            translations=ibz.symmetries.translation_sc,
            weights=ibz.weight_k)

        nproj = self.wfs_qs[0][0].P_ani.layout.size

        spin_k_shape: tuple[int, ...]
        proj_shape: tuple[int, ...]

        if self.collinear:
            spin_k_shape = (self.ncomponents, len(ibz))
            proj_shape = (self.nbands, nproj)
        else:
            spin_k_shape = (len(ibz),)
            proj_shape = (self.nbands, 2, nproj)

        writer.add_array('projections', spin_k_shape + proj_shape, self.dtype)

        for spin in range(self.nspins):
            for k, rank in enumerate(self.rank_k):
                if rank == self.kpt_comm.rank:
                    wfs = self.wfs_qs[self.q_k[k]][spin]
                    P_ani = wfs.P_ani.to_cpu().gather()  # gather atoms
                    if P_ani is not None:
                        P_nI = P_ani.matrix.gather()  # gather bands
                        if self.domain_comm.rank == 0:
                            if rank == 0:
                                writer.fill(P_nI.data.reshape(proj_shape))
                            else:
                                self.kpt_comm.send(P_nI.data, 0)
                elif self.kpt_comm.rank == 0:
                    data = np.empty(proj_shape, self.dtype)
                    self.kpt_comm.receive(data, rank)
                    writer.fill(data)

        if skip_wfs:
            return

        xshape = self.get_max_shape(global_shape=True)
        shape = spin_k_shape + (self.nbands,) + xshape

        c = Bohr**-1.5
        if isinstance(wfs, LCAOWaveFunctions):
            c = 1

        for spin in range(self.nspins):
            for k, rank in enumerate(self.rank_k):
                if rank == self.kpt_comm.rank:
                    wfs = self.wfs_qs[self.q_k[k]][spin]
                    coef_nX = wfs.gather_wave_function_coefficients()
                    if coef_nX is not None:
                        if rank == 0:
                            if spin == 0 and k == 0:
                                writer.add_array('coefficients',
                                                 shape, dtype=coef_nX.dtype)
                            # For PW-mode, we may need to zero-padd the
                            # plane-wave coefficient up to the maximum
                            # for all k-points:
                            n = shape[-1] - coef_nX.shape[-1]
                            if n != 0:
                                coef_nX = np.pad(coef_nX, ((0, 0), (0, n)))
                            writer.fill(coef_nX * c)
                        else:
                            self.kpt_comm.send(coef_nX, 0)
                elif self.kpt_comm.rank == 0:
                    if coef_nX is not None:
                        self.kpt_comm.receive(coef_nX, rank)
                        writer.fill(coef_nX * c)

    def write_summary(self, log):
        fl = self.fermi_levels * Ha
        if len(fl) == 1:
            log(f'\nFermi level: {fl[0]:.3f} eV')
        else:
            log(f'\nFermi levels: {fl[0]:.3f}, {fl[1]:.3f} eV')

        ibz = self.ibz

        eig_skn, occ_skn = self.get_all_eigs_and_occs()

        if not self.is_master():
            return

        eig_skn *= Ha

        D = self.spin_degeneracy
        nbands = eig_skn.shape[2]

        for k, (x, y, z) in enumerate(ibz.kpt_kc):
            if k == 3:
                log(f'(only showing first 3 out of {len(ibz)} k-points)')
                break

            log(f'\nkpt = [{x:.3f}, {y:.3f}, {z:.3f}], '
                f'weight = {ibz.weight_k[k]:.3f}:')

            if self.nspins == 1:
                skipping = False
                log(f'  Band      eig [eV]   occ [0-{D}]')
                eig_n = eig_skn[0, k]
                n0 = (eig_n < fl[0]).sum() - 0.5
                for n, (e, f) in enumerate(zip(eig_n, occ_skn[0, k])):
                    # First, last and +-8 bands window around fermi level:
                    if n == 0 or abs(n - n0) < 8 or n == nbands - 1:
                        log(f'  {n:4} {e:13.3f}   {D * f:9.3f}')
                        skipping = False
                    else:
                        if not skipping:
                            log('   ...')
                            skipping = True
            else:
                log('  Band      eig [eV]   occ [0-1]'
                    '      eig [eV]   occ [0-1]')
                for n, (e1, f1, e2, f2) in enumerate(zip(eig_skn[0, k],
                                                         occ_skn[0, k],
                                                         eig_skn[1, k],
                                                         occ_skn[1, k])):
                    log(f'  {n:4} {e1:13.3f}   {f1:9.3f}'
                        f'    {e2:10.3f}   {f2:9.3f}')

        try:
            log()
            bandgap(eigenvalues=eig_skn,
                    efermi=fl[0],
                    output=log.fd,
                    kpts=ibz.kpt_kc)
        except ValueError:
            # Maybe we only have the occupied bands and no empty bands
            pass

    def make_sure_wfs_are_read_from_gpw_file(self):
        for wfs in self:
            psit_nX = getattr(wfs, 'psit_nX', None)
            if psit_nX is None:
                return
            if hasattr(psit_nX.data, 'fd'):
                psit_nX.data = psit_nX.data[:]  # read

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        """Return HOMO and LUMO eigenvalues."""
        if self.ncomponents == 1:
            N = 2
            assert spin != 1
            spin = 0
        elif self.ncomponents == 2:
            N = 2
            if spin is None:
                h0, l0 = self.get_homo_lumo(0)
                h1, l1 = self.get_homo_lumo(1)
                return np.array([max(h0, h1), min(l0, l1)])
        else:
            N = 1
            assert spin != 1
            spin = 0

        n = int(round(self.nelectrons)) // N
        assert N * n == self.nelectrons
        homo = self.kpt_comm.max(max(wfs_s[spin].eig_n[n - 1]
                                     for wfs_s in self.wfs_qs))
        lumo = self.kpt_comm.min(min(wfs_s[spin].eig_n[n]
                                     for wfs_s in self.wfs_qs))

        return np.array([homo, lumo])
