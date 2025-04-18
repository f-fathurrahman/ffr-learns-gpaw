from __future__ import annotations

import numpy as np
from my_gpaw.core.atom_arrays import AtomArrays
from my_gpaw.core.matrix import Matrix, create_distribution
from my_gpaw.core.plane_waves import (PlaneWaveAtomCenteredFunctions,
                                   PlaneWaveExpansions, PlaneWaves)
from my_gpaw.core.uniform_grid import UniformGridFunctions
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from my_gpaw.typing import Array2D
from my_gpaw.new.ibzwfs import IBZWaveFunctions
from my_gpaw.new.wave_functions import WaveFunctions
from my_gpaw.new.potential import Potential
from my_gpaw.new.smearing import OccupationNumberCalculator


def pw_matrix(pw: PlaneWaves,
              pt_aiG: PlaneWaveAtomCenteredFunctions,
              dH_aii: AtomArrays,
              dS_aii: list[Array2D],
              vt_R: UniformGridFunctions,
              comm) -> tuple[Matrix, Matrix]:
    """Calculate H and S matrices in plane-wave basis.

    :::

                 _ _     _ _
            /  -iG.r ~  iG.r _
      O   = | e      O e    dr
       GG'  /

    :::

      ~   ^   ~ _    _ _     --- ~a _ _a    a  ~  _  _a
      H = T + v(r) δ(r-r') + <   p (r-R ) ΔH   p (r'-R )
                             ---  i         ij  j
                             aij

    :::

      ~     _ _     --- ~a _ _a    a  ~  _  _a
      S = δ(r-r') + <   p (r-R ) ΔS   p (r'-R )
                    ---  i         ij  j
                    aij
    """
    assert pw.dtype == complex
    npw = pw.shape[0]
    dist = create_distribution(npw, npw, comm, -1, 1)
    H_GG = dist.matrix(complex)
    S_GG = dist.matrix(complex)
    G1, G2 = dist.my_row_range()

    x_G = pw.empty()
    assert isinstance(x_G, PlaneWaveExpansions)  # Fix this!
    x_R = vt_R.desc.new(dtype=complex).zeros()
    assert isinstance(x_R, UniformGridFunctions)  # Fix this!
    dv = pw.dv

    for G in range(G1, G2):
        x_G.data[:] = 0.0
        x_G.data[G] = 1.0
        x_G.ifft(out=x_R)
        x_R.data *= vt_R.data
        x_R.fft(out=x_G)
        H_GG.data[G - G1] = dv * x_G.data

    H_GG.add_to_diagonal(dv * pw.ekin_G)
    S_GG.data[:] = 0.0
    S_GG.add_to_diagonal(dv)

    pt_aiG._lazy_init()
    assert pt_aiG._lfc is not None
    f_GI = pt_aiG._lfc.expand()
    nI = f_GI.shape[1]
    dH_II = np.zeros((nI, nI))
    dS_II = np.zeros((nI, nI))
    I1 = 0
    for a, dH_ii in dH_aii.items():
        dS_ii = dS_aii[a]
        I2 = I1 + len(dS_ii)
        dH_II[I1:I2, I1:I2] = dH_ii
        dS_II[I1:I2, I1:I2] = dS_ii
        I1 = I2

    H_GG.data += (f_GI[G1:G2].conj() @ dH_II) @ f_GI.T
    S_GG.data += (f_GI[G1:G2].conj() @ dS_II) @ f_GI.T

    return H_GG, S_GG


def diagonalize(potential: Potential,
                ibzwfs: IBZWaveFunctions,
                occ_calc: OccupationNumberCalculator,
                nbands: int | None) -> IBZWaveFunctions:
    """Diagonalize hamiltonian in plane-wave basis."""
    vt_sR = potential.vt_sR
    dH_asii = potential.dH_asii

    if nbands is None:
        nbands = min(wfs.array_shape(global_shape=True)[0]
                     for wfs in ibzwfs)
        array = np.array(nbands)
        ibzwfs.kpt_comm.max(array)
        nbands = int(array)

    wfs_qs: list[list[WaveFunctions]] = []
    for wfs_s in ibzwfs.wfs_qs:
        wfs_qs.append([])
        for wfs in wfs_s:
            dS_aii = [setup.dO_ii for setup in wfs.setups]
            assert isinstance(wfs, PWFDWaveFunctions)
            assert isinstance(wfs.pt_aiX, PlaneWaveAtomCenteredFunctions)
            H_GG, S_GG = pw_matrix(wfs.psit_nX.desc,
                                   wfs.pt_aiX,
                                   dH_asii[:, wfs.spin],
                                   dS_aii,
                                   vt_sR[wfs.spin],
                                   wfs.psit_nX.comm)
            eig_n = H_GG.eigh(S_GG, limit=nbands)
            if eig_n[0] < -1000:
                raise RuntimeError(
                    f'Lowest eigenvalue is {eig_n[0]} Hartree. '
                    'You might be suffering from MKL library bug MKLD-11440. '
                    'See issue #241 in GPAW. '
                    'Creashing to prevent corrupted results.')
            psit_nG = wfs.psit_nX.desc.empty(nbands)
            psit_nG.data[:nbands] = H_GG.data[:nbands].conj()
            new_wfs = PWFDWaveFunctions(
                psit_nG,
                wfs.spin,
                wfs.q,
                wfs.k,
                wfs.setups,
                wfs.fracpos_ac,
                wfs.atomdist,
                wfs.weight,
                wfs.ncomponents)
            new_wfs._eig_n = eig_n
            wfs_qs[-1].append(new_wfs)

    new_ibzwfs = IBZWaveFunctions(
        ibzwfs.ibz,
        ibzwfs.nelectrons,
        ibzwfs.ncomponents,
        wfs_qs,
        ibzwfs.kpt_comm)

    new_ibzwfs.calculate_occs(occ_calc)

    return new_ibzwfs
