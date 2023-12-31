import numpy as np
from ase import Atoms
from ase.units import Bohr

from my_gpaw.band_descriptor import BandDescriptor
from my_gpaw.kpt_descriptor import KPointDescriptor
from my_gpaw.new import cached_property, prod
from my_gpaw.new.calculation import DFTCalculation
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from my_gpaw.projections import Projections
from my_gpaw.pw.descriptor import PWDescriptor
from my_gpaw.utilities import pack
from my_gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions


class FakeWFS:
    def __init__(self, calculation: DFTCalculation, atoms: Atoms):
        self.setups = calculation.setups
        self.state = calculation.state
        ibzwfs = self.state.ibzwfs
        self.kd = KPointDescriptor(ibzwfs.ibz.bz.kpt_Kc,
                                   ibzwfs.nspins)
        self.kd.set_symmetry(atoms,
                             ibzwfs.ibz.symmetries.symmetry)
        self.kd.set_communicator(ibzwfs.kpt_comm)
        self.bd = BandDescriptor(ibzwfs.nbands, ibzwfs.band_comm)
        grid = self.state.density.nt_sR.desc
        self.gd = grid._gd
        self.atom_partition = calculation._atom_partition
        self.setups.set_symmetry(ibzwfs.ibz.symmetries.symmetry)
        self.occupations = calculation.scf_loop.occ_calc.occ
        self.nvalence = int(round(ibzwfs.nelectrons))
        assert self.nvalence == ibzwfs.nelectrons
        self.world = calculation.scf_loop.world
        if ibzwfs.fermi_levels is not None:
            self.fermi_levels = ibzwfs.fermi_levels
            if len(self.fermi_levels) == 1:
                self.fermi_level = self.fermi_levels[0]
        self.nspins = ibzwfs.nspins
        self.dtype = ibzwfs.dtype
        wfs = ibzwfs.wfs_qs[0][0]
        self.pd = None
        if isinstance(wfs, PWFDWaveFunctions):
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                self.mode = 'pw'
                self.pd = PWDescriptor(wfs.psit_nX.desc.ecut,
                                       self.gd, self.dtype, self.kd)
                self.pwgrid = grid.new(dtype=self.dtype)
            else:
                self.mode = 'fd'
        else:
            self.mode = 'lcao'

    def _get_wave_function_array(self, u, n, realspace):
        psit_X = self.kpt_u[u].wfs.psit_nX[n]
        if hasattr(psit_X, 'ifft'):
            return psit_X.ifft(grid=self.pwgrid).data
        return psit_X.data

    def get_wave_function_array(self, n, k, s, realspace=True, periodic=False):
        assert realspace
        psit_X = self.kpt_qs[k][s].wfs.psit_nX[n]
        if self.mode == 'pw':
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=periodic)
        else:
            psit_R = psit_X
            if periodic:
                psit_R.multiply_by_eikr(-psit_R.desc.kpt_c)
        return psit_R.data

    def collect_projections(self, k, s):
        return self.kpt_qs[k][s].projections.collect()

    def collect_eigenvalues(self, k, s):
        return self.state.ibzwfs.wfs_qs[k][s].eig_n.copy()

    @cached_property
    def kpt_u(self):
        return [kpt
                for kpt_s in self.kpt_qs
                for kpt in kpt_s]

    @cached_property
    def kpt_qs(self):
        ngpts = prod(self.gd.N_c)
        return [[KPT(wfs, self.atom_partition, ngpts, self.pd)
                 for wfs in wfs_s]
                for wfs_s in self.state.ibzwfs.wfs_qs]


class KPT:
    def __init__(self, wfs, atom_partition, ngpts, pd):
        self.ngpts = ngpts
        self.wfs = wfs
        self.pd = pd
        self.projections = Projections(
            wfs.nbands,
            [I2 - I1 for (a, I1, I2) in wfs.P_ani.layout.myindices],
            atom_partition,
            wfs.P_ani.comm,
            wfs.ncomponents < 4,
            wfs.spin,
            data=wfs.P_ani.data)
        self.eps_n = wfs.eig_n
        self.s = wfs.spin if wfs.ncomponents < 4 else None
        self.k = wfs.k
        self.q = wfs.q
        self.weight = wfs.spin_degeneracy * wfs.weight
        self.f_n = wfs.occ_n * self.weight
        self.P_ani = wfs.P_ani
        if isinstance(wfs, PWFDWaveFunctions):
            self.psit_nX = wfs.psit_nX

    @property
    def psit_nG(self):
        a_nG = self.psit_nX.data
        if a_nG.ndim == 4:
            return a_nG
        return a_nG * self.ngpts

    @cached_property
    def psit(self):
        return PlaneWaveExpansionWaveFunctions(
            self.wfs.nbands, self.pd, self.wfs.dtype,
            self.psit_nX.data * self.ngpts,
            kpt=self.q,
            dist=(),  # self.bd.comm, self.bd.comm.size),
            spin=self.s,
            collinear=self.wfs.ncomponents != 4)


class FakeDensity:
    def __init__(self, calculation: DFTCalculation):
        self.setups = calculation.setups
        self.state = calculation.state
        self.D_asii = self.state.density.D_asii
        self.atom_partition = calculation._atom_partition
        self.interpolate = calculation.pot_calc._interpolate_density
        self.nt_sR = self.state.density.nt_sR
        self.nt_sG = self.nt_sR.data
        self.gd = self.nt_sR.desc._gd
        self.finegd = calculation.pot_calc.fine_grid._gd
        self._densities = calculation.densities()
        self.ncomponents = len(self.nt_sG)
        self.nspins = self.ncomponents % 3

    @cached_property
    def D_asp(self):
        D_asp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                self.atom_partition)
        D_asp.update({a: np.array([pack(D_ii) for D_ii in D_sii])
                      for a, D_sii in self.D_asii.items()})
        return D_asp

    @cached_property
    def nt_sg(self):
        return self.interpolate(self.nt_sR)[0].data

    def interpolate_pseudo_density(self):
        pass

    def get_all_electron_density(self, *, atoms, gridrefinement):
        n_sr = self._densities.all_electron_densities(
            grid_refinement=gridrefinement).scaled(1 / Bohr, Bohr**3)
        return n_sr.data, n_sr.desc._gd


class FakeHamiltonian:
    def __init__(self, calculation: DFTCalculation):
        self.pot_calc = calculation.pot_calc
        self.finegd = self.pot_calc.fine_grid._gd
        self.grid = calculation.state.potential.vt_sR.desc
        self.e_total_free = calculation.results.get('free_energy')
        self.e_xc = calculation.state.potential.energies['xc']
        # self.poisson = calculation.pot_calc.poisson_solver.solver

    def restrict_and_collect(self, vxct_sg):
        fine_grid = self.pot_calc.fine_grid
        vxct_sr = fine_grid.empty(len(vxct_sg))
        vxct_sr.data[:] = vxct_sg
        vxct_sR = self.pot_calc.restrict(vxct_sr)
        return vxct_sR.data

    @property
    def xc(self):
        return self.pot_calc.xc.xc
