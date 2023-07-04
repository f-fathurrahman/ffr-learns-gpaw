from __future__ import annotations

from types import ModuleType

import numpy as np
from my_gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from my_gpaw.core.uniform_grid import UniformGridFunctions
from my_gpaw.mpi import MPIComm, serial_comm
from my_gpaw.new import zip
from my_gpaw.new.potential import Potential
from my_gpaw.setup import Setups
from my_gpaw.typing import Array1D, Array2D, ArrayND


class WaveFunctions:
    bytes_per_band: int
    xp: ModuleType  # numpy or cupy

    def __init__(self,
                 *,
                 setups: Setups,
                 nbands: int,
                 fracpos_ac: Array2D,
                 atomdist: AtomDistribution,
                 spin: int = 0,
                 q: int = 0,
                 k: int = 0,
                 kpt_c=(0.0, 0.0, 0.0),
                 weight: float = 1.0,
                 ncomponents: int = 1,
                 dtype=float,
                 domain_comm: MPIComm = serial_comm,
                 band_comm: MPIComm = serial_comm):
        """"""
        assert spin < ncomponents

        self.spin = spin
        self.q = q
        self.k = k
        self.setups = setups
        self.weight = weight
        self.ncomponents = ncomponents
        self.dtype = dtype
        self.kpt_c = kpt_c
        self.fracpos_ac = fracpos_ac
        self.atomdist = atomdist
        self.domain_comm = domain_comm
        self.band_comm = band_comm
        self.nbands = nbands

        assert domain_comm.size == atomdist.comm.size

        self.nspins = ncomponents % 3
        self.spin_degeneracy = ncomponents % 2 + 1

        self._P_ani: AtomArrays | None = None

        self._eig_n: Array1D | None = None
        self._occ_n: Array1D | None = None

    def __repr__(self):
        dc = f'{self.domain_comm.rank}/{self.domain_comm.size}'
        bc = f'{self.band_comm.rank}/{self.band_comm.size}'
        return (f'{self.__class__.__name__}(nbands={self.nbands}, '
                f'spin={self.spin}, q={self.q}, k={self.k}, '
                f'weight={self.weight}, kpt_c={self.kpt_c}, '
                f'ncomponents={self.ncomponents}, dtype={self.dtype} '
                f'domain_comm={dc}, band_comm={bc})')

    def array_shape(self, global_shape: bool = False) -> tuple[int, ...]:
        raise NotImplementedError

    def add_to_density(self,
                       nt_sR: UniformGridFunctions,
                       D_asii: AtomArrays) -> None:
        raise NotImplementedError

    def orthonormalize(self, work_array_nX: ArrayND = None):
        raise NotImplementedError

    def collect(self,
                n1: int = 0,
                n2: int = 0) -> WaveFunctions | None:
        raise NotImplementedError

    @property
    def eig_n(self) -> Array1D:
        if self._eig_n is None:
            raise ValueError
        return self._eig_n

    @property
    def occ_n(self) -> Array1D:
        if self._occ_n is None:
            raise ValueError
        return self._occ_n

    @property
    def myeig_n(self):
        assert self.band_comm.size == 1
        return self.eig_n

    @property
    def myocc_n(self):
        assert self.band_comm.size == 1
        return self.occ_n

    @property
    def P_ani(self) -> AtomArrays:
        assert self._P_ani is not None
        return self._P_ani

    def add_to_atomic_density_matrices(self,
                                       occ_n,
                                       D_asii: AtomArrays) -> None:
        xp = D_asii.layout.xp
        occ_n = xp.asarray(occ_n)
        if self.ncomponents < 4:
            P_ani = self.P_ani
            for D_sii, P_ni in zip(D_asii.values(), P_ani.values()):
                D_sii[self.spin] += xp.einsum('ni, n, nj -> ij',
                                              P_ni.conj(), occ_n, P_ni).real
        else:
            for D_xii, P_nsi in zip(D_asii.values(), self.P_ani.values()):
                D_ssii = xp.einsum('nsi, n, nzj -> szij',
                                   P_nsi.conj(), occ_n, P_nsi)
                D_xii[0] += (D_ssii[0, 0] + D_ssii[1, 1]).real
                D_xii[1] += 2 * D_ssii[0, 1].real
                D_xii[2] += 2 * D_ssii[0, 1].imag
                D_xii[3] += (D_ssii[0, 0] - D_ssii[1, 1]).real

    def send(self, kpt_comm, rank):
        raise NotImplementedError

    def receive(self, kpt_comm, rank):
        raise NotImplementedError

    def force_contribution(self, potential: Potential, F_av: Array2D):
        raise NotImplementedError

    def gather_wave_function_coefficients(self) -> np.ndarray | None:
        raise NotImplementedError
