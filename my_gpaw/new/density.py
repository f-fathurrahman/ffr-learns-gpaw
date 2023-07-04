from __future__ import annotations
from math import sqrt, pi
import numpy as np
from ase.units import Bohr
from my_gpaw.typing import Vector
from my_gpaw.core.atom_centered_functions import AtomArraysLayout
from my_gpaw.utilities import unpack2, unpack
from my_gpaw.core.atom_arrays import AtomArrays
from my_gpaw.core.uniform_grid import UniformGridFunctions
from my_gpaw.gpu import as_xp
from my_gpaw.new import zip
from my_gpaw.core.plane_waves import PlaneWaves


class Density:
    def __init__(self,
                 nt_sR: UniformGridFunctions,
                 D_asii: AtomArrays,
                 charge: float,
                 delta_aiiL: list,
                 delta0_a,
                 N0_aii,
                 l_aj):
        self.nt_sR = nt_sR
        self.D_asii = D_asii
        self.delta_aiiL = delta_aiiL
        self.delta0_a = delta0_a
        self.N0_aii = N0_aii
        self.l_aj = l_aj
        self.charge = charge

        self.ncomponents = nt_sR.dims[0]
        self.ndensities = self.ncomponents % 3
        self.collinear = self.ncomponents != 4
        self.natoms = len(delta0_a)

    def __repr__(self):
        return f'Density({self.nt_sR}, {self.D_asii}, charge={self.charge})'

    def __str__(self) -> str:
        return (f'density:\n'
                f'  components: {self.ncomponents}\n'
                f'  grid points: {self.nt_sR.desc.size}\n'
                f'  charge: {self.charge}  # |e|\n')

    def new(self, grid):
        old_grid = self.nt_sR.desc
        nt_sR = grid.empty(self.ncomponents, xp=self.nt_sR.xp)
        ecut = 0.999 * min(grid.ecut_max(), old_grid.ecut_max())
        pw = PlaneWaves(ecut=ecut, cell=old_grid.cell, comm=grid.comm)
        for nt_R, old_nt_R in zip(nt_sR, self.nt_sR):
            old_nt_R.fft(pw=pw).ifft(out=nt_R)

        return Density(nt_sR,
                       self.D_asii,
                       self.charge,
                       self.delta_aiiL,
                       self.delta0_a,
                       self.N0_aii,
                       self.l_aj)

    def calculate_compensation_charge_coefficients(self) -> AtomArrays:
        xp = self.D_asii.layout.xp
        ccc_aL = AtomArraysLayout(
            [delta_iiL.shape[2] for delta_iiL in self.delta_aiiL],
            atomdist=self.D_asii.layout.atomdist,
            xp=xp).empty()

        for a, D_sii in self.D_asii.items():
            Q_L = xp.einsum('sij, ijL -> L',
                            D_sii[:self.ndensities], self.delta_aiiL[a])
            Q_L[0] += self.delta0_a[a]
            ccc_aL[a] = Q_L

        return ccc_aL

    def normalize(self):
        comp_charge = 0.0
        xp = self.D_asii.layout.xp
        for a, D_sii in self.D_asii.items():
            comp_charge += xp.einsum('sij, ij ->',
                                     D_sii[:self.ndensities],
                                     self.delta_aiiL[a][:, :, 0])
            comp_charge += self.delta0_a[a]
        # comp_charge could be cupy.ndarray:
        comp_charge = float(comp_charge) * sqrt(4 * pi)
        comp_charge = self.nt_sR.desc.comm.sum(comp_charge)
        charge = comp_charge + self.charge
        pseudo_charge = self.nt_sR[:self.ndensities].integrate().sum()
        if pseudo_charge != 0.0:
            x = -charge / pseudo_charge
            self.nt_sR.data *= x

    def update(self, nct_R, ibzwfs):
        self.nt_sR.data[:] = 0.0
        self.D_asii.data[:] = 0.0
        ibzwfs.add_to_density(self.nt_sR, self.D_asii)
        self.nt_sR.data[:self.ndensities] += nct_R.data
        self.symmetrize(ibzwfs.ibz.symmetries)

    def symmetrize(self, symmetries):
        self.nt_sR.symmetrize(symmetries.rotation_scc,
                              symmetries.translation_sc)
        xp = self.nt_sR.xp
        D_asii = self.D_asii.gather(broadcast=True, copy=True)
        for a1, D_sii in self.D_asii.items():
            D_sii[:] = 0.0
            rotation_sii = symmetries.rotations(self.l_aj[a1], xp)
            for a2, rotation_ii in zip(symmetries.a_sa[:, a1],
                                       rotation_sii):
                D_sii += xp.einsum('ij, sjk, lk -> sil',
                                   rotation_ii, D_asii[a2], rotation_ii)
        self.D_asii.data *= 1.0 / len(symmetries)

    def move(self, delta_nct_R):
        self.nt_sR.data[:self.ndensities] += delta_nct_R.data

    @classmethod
    def from_superposition(cls,
                           grid,
                           nct_R,
                           atomdist,
                           setups,
                           basis_set,
                           magmom_av,
                           ncomponents,
                           charge=0.0,
                           hund=False):
        f_asi = {a: atomic_occupation_numbers(setup,
                                              magmom_v,
                                              ncomponents,
                                              hund,
                                              charge / len(setups))
                 for a, (setup, magmom_v) in enumerate(zip(setups, magmom_av))}

        nt_sR = nct_R.desc.zeros(ncomponents)
        basis_set.add_to_density(nt_sR.data, f_asi)
        ndensities = ncomponents % 3
        nt_sR.data[:ndensities] += nct_R.to_xp(np).data

        atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                              for setup in setups],
                                             atomdist=atomdist)
        D_asii = atom_array_layout.empty(ncomponents)
        for a, D_sii in D_asii.items():
            D_sii[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a]))

        xp = nct_R.xp
        return cls.from_data_and_setups(nt_sR.to_xp(xp),
                                        D_asii.to_xp(xp),
                                        charge,
                                        setups)

    @classmethod
    def from_data_and_setups(cls,
                             nt_sR,
                             D_asii,
                             charge,
                             setups):
        xp = nt_sR.xp
        return cls(nt_sR,
                   D_asii,
                   charge,
                   [xp.asarray(setup.Delta_iiL) for setup in setups],
                   [setup.Delta0 for setup in setups],
                   [unpack(setup.N0_p) for setup in setups],
                   [setup.l_j for setup in setups])

    def calculate_dipole_moment(self, fracpos_ac):
        dip_v = np.zeros(3)
        ccc_aL = self.calculate_compensation_charge_coefficients()
        ccc_aL = ccc_aL.to_cpu()
        pos_av = fracpos_ac @ self.nt_sR.desc.cell_cv
        for a, ccc_L in ccc_aL.items():
            c = ccc_L[0]
            dip_v -= c * (4 * pi)**0.5 * pos_av[a]
            if len(ccc_L) > 1:
                y, z, x = ccc_L[1:4]
                dip_v -= np.array([x, y, z]) * (4 * pi / 3)**0.5
        self.nt_sR.desc.comm.sum(dip_v)
        for nt_R in self.nt_sR:
            dip_v -= as_xp(nt_R.moment(), np)
        return dip_v

    def calculate_magnetic_moments(self):
        magmom_av = np.zeros((self.natoms, 3))
        magmom_v = np.zeros(3)
        domain_comm = self.nt_sR.desc.comm

        if self.ncomponents == 2:
            for a, D_sii in self.D_asii.items():
                M_ii = D_sii[0] - D_sii[1]
                magmom_av[a, 2] = np.einsum('ij, ij ->', M_ii, self.N0_aii[a])
                magmom_v[2] += (np.einsum('ij, ij ->', M_ii,
                                          self.delta_aiiL[a][:, :, 0]) *
                                sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)

            M_s = self.nt_sR.integrate()
            magmom_v[2] += M_s[0] - M_s[1]

        elif self.ncomponents == 4:
            for a, D_sii in self.D_asii.items():
                M_vii = D_sii[1:4]
                magmom_av[a] = np.einsum('vij, ij -> v',
                                         M_vii, self.N0_aii[a])
                magmom_v += (np.einsum('vij, ij -> v', M_vii,
                                       self.delta_aiiL[a][:, :, 0]) *
                             sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)
            magmom_v += self.nt_sR.integrate()[1:]

        return magmom_v, magmom_av

    def write(self, writer):
        D_asp = self.D_asii.to_cpu().to_lower_triangle().gather()
        nt_sR = self.nt_sR.to_xp(np).gather()
        if D_asp is None:
            return

        writer.write(
            density=nt_sR.data * Bohr**-3,
            atomic_density_matrices=D_asp.data)


def atomic_occupation_numbers(setup,
                              magmom_v: Vector,
                              ncomponents: int,
                              hund: bool = False,
                              charge: float = 0.0):
    M = np.linalg.norm(magmom_v)
    nspins = min(ncomponents, 2)
    f_si = setup.calculate_initial_occupation_numbers(
        M, hund, charge=charge, nspins=nspins)

    if ncomponents == 1:
        pass
    elif ncomponents == 2:
        if magmom_v[2] < 0:
            f_si = f_si[::-1].copy()
    else:
        f_i = f_si.sum(0)
        fm_i = f_si[0] - f_si[1]
        f_si = np.zeros((4, len(f_i)))
        f_si[0] = f_i
        if M > 0:
            f_si[1:] = np.asarray(magmom_v)[:, np.newaxis] / M * fm_i

    return f_si
