from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from my_gpaw.core.atom_arrays import AtomArrays
from my_gpaw.gpu import synchronize, as_xp
from my_gpaw.new.calculation import DFTState
from my_gpaw.new.ibzwfs import IBZWaveFunctions
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from my_gpaw.typing import Array2D

if TYPE_CHECKING:
    from my_gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator


def calculate_stress(pot_calc: PlaneWavePotentialCalculator,
                     state: DFTState,
                     vt_g,
                     nt_g) -> Array2D:
    assert state.ibzwfs.domain_comm.size == 1
    assert state.ibzwfs.band_comm.size == 1
    world = state.ibzwfs.kpt_comm

    xc = pot_calc.xc

    if xc.xc.orbital_dependent and xc.type != 'MGGA':
        raise NotImplementedError('Calculation of stress tensor is not ' +
                                  'implemented for orbital-dependent ' +
                                  'XC functionals such as ' + xc.name)
    assert xc.type != 'MGGA'
    assert not xc.no_forces

    s_vv = get_wfs_stress(state.ibzwfs, state.potential.dH_asii)

    nt_sr = pot_calc._interpolate_density(state.density.nt_sR)[0]
    xp = nt_sr.xp
    s_vv += as_xp(
        pot_calc.xc.xc.stress_tensor_contribution(as_xp(nt_sr.data, np)), xp)

    vHt_h = state.vHt_x
    assert vHt_h is not None
    pw = vHt_h.desc
    G_Gv = xp.asarray(pw.G_plus_k_Gv)
    vHt2_hz = vHt_h.data.view(float).reshape((len(G_Gv), 2))**2
    s_vv += (xp.einsum('Gz, Gv, Gw -> vw', vHt2_hz, G_Gv, G_Gv) *
             pw.dv / (2 * np.pi))

    Q_aL = state.density.calculate_compensation_charge_coefficients()
    s_vv += pot_calc.ghat_aLh.stress_tensor_contribution(vHt_h, Q_aL)

    s_vv -= xp.eye(3) * pot_calc.e_stress
    s_vv += pot_calc.vbar_ag.stress_tensor_contribution(nt_g)
    s_vv += pot_calc.nct_ag.stress_tensor_contribution(vt_g)

    # s_vv += wfs.dedepsilon * np.eye(3)

    s_vv = as_xp(s_vv, np)

    vol = pw.volume
    s_vv = 0.5 / vol * (s_vv + s_vv.T)

    # Symmetrize:
    sigma_vv = np.zeros((3, 3))
    cell_cv = pw.cell_cv
    icell_cv = pw.icell
    rotation_scc = state.ibzwfs.ibz.symmetries.rotation_scc
    for U_cc in rotation_scc:
        M_vv = (icell_cv.T @ (U_cc @ cell_cv)).T
        sigma_vv += M_vv.T @ s_vv @ M_vv
    sigma_vv /= len(rotation_scc)

    # Make sure all agree on the result (redundant calculation on
    # different cores involving BLAS might give slightly different
    # results):
    world.broadcast(sigma_vv, 0)
    return sigma_vv


def get_wfs_stress(ibzwfs: IBZWaveFunctions,
                   dH_asii: AtomArrays) -> Array2D:
    xp = ibzwfs.xp
    sigma_vv = xp.zeros((3, 3))
    for wfs in ibzwfs:
        assert isinstance(wfs, PWFDWaveFunctions)
        occ_n = xp.asarray(wfs.weight * wfs.spin_degeneracy * wfs.myocc_n)
        sigma_vv += get_kinetic_stress(wfs, occ_n)
        sigma_vv += get_paw_stress(wfs, dH_asii, occ_n)
    if xp is not np:
        synchronize()
    ibzwfs.kpt_comm.sum(sigma_vv)
    return sigma_vv


def get_kinetic_stress(wfs: PWFDWaveFunctions,
                       occ_n) -> Array2D:
    psit_nG = wfs.psit_nX
    pw = psit_nG.desc
    xp = psit_nG.xp
    psit_nGz = psit_nG.data.view(float).reshape(psit_nG.data.shape + (2,))
    psit2_G = xp.einsum('n, nGz, nGz -> G', occ_n, psit_nGz, psit_nGz)
    Gk_Gv = xp.asarray(pw.G_plus_k_Gv)
    sigma_vv = xp.einsum('G, Gv, Gw -> vw', psit2_G, Gk_Gv, Gk_Gv)
    x = pw.dv
    if pw.dtype == float:
        x *= 2
    return -x * sigma_vv


def get_paw_stress(wfs: PWFDWaveFunctions,
                   dH_asii: AtomArrays,
                   occ_n) -> Array2D:
    xp = wfs.xp
    eig_n1 = xp.asarray(wfs.eig_n[:, None])
    a_ani = {}
    s = 0.0
    for a, P_ni in wfs.P_ani.items():
        Pf_ni = P_ni * occ_n[:, None]
        dH_ii = dH_asii[a][wfs.spin]
        dS_ii = xp.asarray(wfs.setups[a].dO_ii)
        a_ni = (Pf_ni @ dH_ii - Pf_ni * eig_n1 @ dS_ii)
        s += xp.vdot(P_ni, a_ni)
        a_ani[a] = 2 * a_ni.conj()
    s_vv = wfs.pt_aiX.stress_tensor_contribution(wfs.psit_nX, a_ani)
    return s_vv - float(s.real) * xp.eye(3)
