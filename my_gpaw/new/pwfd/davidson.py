from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from ase.units import Ha
from my_gpaw.core.arrays import DistributedArrays as DA
from my_gpaw.core.atom_centered_functions import AtomArrays as AA
from my_gpaw.core.matrix import Matrix
from my_gpaw.gpu import as_xp
from my_gpaw.mpi import broadcast_float
from my_gpaw.new import zip
from my_gpaw.new.calculation import DFTState
from my_gpaw.new.eigensolver import Eigensolver
from my_gpaw.new.hamiltonian import Hamiltonian
from my_gpaw.new.ibzwfs import IBZWaveFunctions
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from my_gpaw.typing import Array1D, Array2D
from my_gpaw.utilities.blas import axpy
from my_gpaw.yml import obj2yaml as o2y

AAFunc = Callable[[AA, AA], AA]


class Davidson(Eigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 niter=2,
                 blocksize=10,
                 converge_bands='occupied',
                 scalapack_parameters=None):
        self.niter = niter
        self.converge_bands = converge_bands

        self.H_NN = None
        self.S_NN = None
        self.M_nn = None
        self.work_arrays: np.ndarray | None = None

        self.preconditioner = preconditioner_factory(blocksize)

    def __str__(self):
        return o2y(dict(name='Davidson',
                        niter=self.niter,
                        converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        # First time: allocate work-arrays
        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        xp = wfs.psit_nX.xp
        B = ibzwfs.nbands
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm
        shape = ibzwfs.get_max_shape()
        shape = (2, B) + shape
        dtype = wfs.psit_nX.data.dtype
        self.work_arrays = xp.empty(shape, dtype)

        dtype = wfs.psit_nX.desc.dtype
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
            self.S_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
        else:
            self.H_NN = self.S_NN = Matrix(0, 0)

        self.M_nn = Matrix(B, B, dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                   ~     ~ ~
              R = (H - ε S)ψ
               n        n   n
        """

        print("Pass here in Davidson(Eigensolver) iterate")

        if self.work_arrays is None:
            self._initialize(state.ibzwfs)

        assert self.M_nn is not None

        dS = state.ibzwfs.wfs_qs[0][0].setups.overlap_correction
        dH = state.potential.dH
        Ht = partial(hamiltonian.apply, state.potential.vt_sR)
        ibzwfs = state.ibzwfs
        error = 0.0

        weight_un = calculate_weights(self.converge_bands, ibzwfs)

        for wfs, weight_n in zip(ibzwfs, weight_un):
            e = self.iterate1(wfs, Ht, dH, dS, weight_n)
            error += wfs.weight * e
        return ibzwfs.kpt_comm.sum(float(error)) * ibzwfs.spin_degeneracy

    def iterate1(self, wfs, Ht, dH, dS, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        psit2_nX = psit_nX.new(data=self.work_arrays[0])
        psit3_nX = psit_nX.new(data=self.work_arrays[1])

        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn
        assert band_comm.size == 1

        if domain_comm.rank == 0:
            eig_N[:B] = xp.asarray(wfs.eig_n)

        def me(a, b, function=None):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True)

        Ht = partial(Ht, out=residual_nX, spin=wfs.spin)
        dH = partial(dH, spin=wfs.spin)

        calculate_residuals(residual_nX, dH, dS, wfs, P2_ani, P3_ani)

        def copy(C_nn: Array2D) -> None:
            domain_comm.sum(M_nn.data, 0)
            if domain_comm.rank == 0:
                M_nn.redist(M0_nn)
                if band_comm.rank == 0:
                    C_nn[:] = M0_nn.data

        for i in range(self.niter):
            if i == self.niter - 1:  # last iteration
                # Calulate error before we destroy residuals:
                if weight_n is None:
                    error = np.inf
                else:
                    error = weight_n @ as_xp(residual_nX.norm2(), np)
                    if wfs.ncomponents == 4:
                        error = error.sum()

            self.preconditioner(psit_nX, residual_nX, out=psit2_nX)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)

            # <psi2 | H | psi2>
            me(psit2_nX, psit2_nX, function=Ht)
            dH(P2_ani, out_ani=P3_ani)
            P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(H_NN.data[B:, B:])

            # <psi2 | H | psi>
            me(residual_nX, psit_nX)
            P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
            copy(H_NN.data[B:, :B])

            # <psi2 | S | psi2>
            me(psit2_nX, psit2_nX)
            dS(P2_ani, out_ani=P3_ani)
            P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(S_NN.data[B:, B:])

            # <psi2 | S | psi>
            me(psit2_nX, psit_nX)
            P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
            copy(S_NN.data[B:, :B])

            if is_domain_band_master:
                H_NN.data[:B, :B] = xp.diag(eig_N[:B])
                S_NN.data[:B, :B] = xp.eye(B)
                eig_N[:] = H_NN.eigh(S_NN)
                wfs._eig_n = as_xp(eig_N[:B], np)
            if domain_comm.rank == 0:
                band_comm.broadcast(wfs.eig_n, 0)
            domain_comm.broadcast(wfs.eig_n, 0)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    # M0_nn.data[:] = H_NN.data[:B, :B]
                    M0_nn.data[:] = H_NN.data[:B, :B].T
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit_nX, opa='C', out=residual_nX)
            M_nn.multiply(P_ani, opa='C', out=P3_ani)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    # M0_nn.data[:] = H_NN.data[B:, :B]
                    M0_nn.data[:] = H_NN.data[:B, B:].T
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit2_nX, opa='C', beta=1.0, out=residual_nX)
            M_nn.multiply(P2_ani, opa='C', beta=1.0, out=P3_ani)
            psit_nX.data[:] = residual_nX.data
            P_ani, P3_ani = P3_ani, P_ani
            wfs._P_ani = P_ani

            if i < self.niter - 1:
                Ht(psit_nX)
                calculate_residuals(residual_nX, dH, dS, wfs, P2_ani, P3_ani)

        return error


def calculate_residuals(residual_nX: DA,
                        dH: AAFunc,
                        dS: AAFunc,
                        wfs: PWFDWaveFunctions,
                        P1_ani: AA,
                        P2_ani: AA) -> None:

    eig_n = wfs.myeig_n
    xp = residual_nX.xp
    if xp is np:
        for r, e, p in zip(residual_nX.data, wfs.myeig_n, wfs.psit_nX.data):
            axpy(-e, p, r)
    else:
        eig_n = xp.asarray(eig_n)
        for r, e, p in zip(residual_nX.data, eig_n, wfs.psit_nX.data):
            r -= p * e

    dH(wfs.P_ani, P1_ani)
    if wfs.ncomponents < 4:
        subscripts = 'nI, n -> nI'
    else:
        subscripts = 'nsI, n -> nsI'
    if xp is np:
        np.einsum(subscripts, wfs.P_ani.data, eig_n, out=P2_ani.data)
    else:
        P2_ani.data[:] = xp.einsum(subscripts, wfs.P_ani.data, eig_n)
    dS(P2_ani, P2_ani)
    P1_ani.data -= P2_ani.data
    wfs.pt_aiX.add_to(residual_nX, P1_ani)


def calculate_weights(converge_bands: int | str,
                      ibzwfs: IBZWaveFunctions) -> list[Array1D | None]:
    """Calculate convergence weights for all eigenstates."""
    assert ibzwfs.band_comm.size == 1, 'not implemented!'
    weight_un = []
    nu = len(ibzwfs.wfs_qs) * ibzwfs.nspins
    nbands = ibzwfs.nbands

    if converge_bands == 'occupied':
        # Converge occupied bands:
        for wfs in ibzwfs:
            try:
                # Methfessel-Paxton or cold-smearing distributions can give
                # negative occupation numbers - so we take the absolute value:
                weight_n = np.abs(wfs.occ_n)
            except ValueError:
                # No eigenvalues yet:
                return [None] * nu
            weight_un.append(weight_n)
        return weight_un

    if converge_bands == 'all':
        return [np.ones(nbands)] * nu

    if isinstance(converge_bands, int):
        # Converge fixed number of bands:
        n = converge_bands

        for wfs in ibzwfs:
            weight_n = np.zeros(nbands)
            if n < 0:
                n += nbands
            weight_n[:n] = 1.0
            weight_un.append(weight_n)
        return weight_un

    # Converge states with energy up to CBM + delta:
    assert converge_bands.startswith('CBM+')
    delta = float(converge_bands[4:]) / Ha

    if ibzwfs.fermi_levels is None:
        return [None] * nu

    efermi = np.mean(ibzwfs.fermi_levels)

    # Find CBM:
    cbm = np.inf
    nocc_u = np.empty(nu, int)
    for u, wfs in enumerate(ibzwfs):
        n = (wfs.eig_n < efermi).sum()  # number of occupied bands
        nocc_u[u] = n
        if n < nbands:
            cbm = min(cbm, wfs.eig_n[n])

    # If all k-points don't have the same number of occupied bands,
    # then it's a metal:
    n0 = int(broadcast_float(float(nocc_u[0]), ibzwfs.kpt_comm))
    metal = bool(ibzwfs.kpt_comm.sum_scalar(float((nocc_u != n0).any())))
    if metal:
        cbm = efermi
    else:
        cbm = ibzwfs.kpt_comm.min_scalar(cbm)

    ecut = cbm + delta

    for wfs in ibzwfs:
        weight_n = (wfs.eig_n < ecut).astype(float)
        if wfs.eig_n[-1] < ecut:
            # We don't have enough bands!
            weight_n[:] = np.inf
        weight_un.append(weight_n)

    return weight_un
