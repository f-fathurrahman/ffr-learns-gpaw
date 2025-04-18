import numpy as np
import pytest
from my_gpaw import GPAW
from my_gpaw.response.ibz2bz import IBZ2BZMaps
import gpaw.mpi as mpi
from my_gpaw.test.conftest import response_band_cutoff


@pytest.mark.later
@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('gs', ['fancy_si_pw',
                                'al_pw',
                                'fe_pw',
                                'co_pw',
                                'gaas_pw',
                                'v2br4_pw',
                                'srvo3_pw'])
def test_ibz2bz(in_tmp_dir, gpw_files, gs):
    """ Tests gpaw.response.ibz2bz.py
    Tests functionalities to take wavefunction and projections from
    ibz to full bz by comparing calculations with and without symmetry.
    """

    atol = 5e-03  # Tolerance when comparing wfs and projections
    atol_eig = 1e-04  # Tolerance when comparing eigenvalues
    atol_deg = 5e-3  # Tolerance for checking degenerate states
    
    # Loading calc with symmetry
    calc = GPAW(gpw_files[gs + '_wfs'],
                communicator=mpi.serial_comm)
    wfs = calc.wfs
    nconv = response_band_cutoff[gs + '_wfs']

    # setting basic stuff
    nbands = wfs.bd.nbands if nconv == -1 else nconv
    nbzk = wfs.kd.nbzkpts
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs + '_nosym_wfs'],
                      communicator=mpi.serial_comm)
    wfs_nosym = calc_nosym.wfs

    # Check some basic stuff
    assert wfs_nosym.kd.nbzkpts == wfs_nosym.kd.nibzkpts

    # Loop over spins and k-points
    for s in range(wfs.nspins):
        for K in range(nbzk):
            ik = wfs.kd.bz2ibz_k[K]  # IBZ k-point

            # Check so that BZ kpoints are the same
            assert np.allclose(wfs.kd.bzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            assert np.allclose(wfs_nosym.kd.ibzk_kc[K], wfs_nosym.kd.bzk_kc[K])

            # Get data for calc without symmetry at BZ kpt K
            eps_n_nosym, ut_nR_nosym, proj_nosym, dO_aii_nosym = \
                get_ibz_data_from_wfs(wfs_nosym, nbands, K, s)

            # Get data for calc with symmetry at ibz kpt ik
            eps_n, ut_nR, proj, dO_aii = get_ibz_data_from_wfs(wfs,
                                                               nbands,
                                                               ik, s)

            # Map projections and u:s from ik to K
            proj_sym = ibz2bz[K].map_projections(proj)
            ut_nR_sym = np.array([ibz2bz[K].map_pseudo_wave_to_BZ(
                ut_nR[n]) for n in range(nbands)])

            # Check so that eigenvalues are the same
            assert np.allclose(eps_n[:nbands],
                               eps_n_nosym[:nbands],
                               atol=atol_eig)

            # Check so that overlaps are the same for both calculations
            assert equal_dicts(dO_aii,
                               dO_aii_nosym,
                               atol)

            # Here starts the actual test
            # Loop over all bands
            n = 0
            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands, atol_deg)
                if dim == 1:
                    
                    # First check untransformed quantities for ibz k-points
                    if np.allclose(wfs.kd.bzk_kc[K],
                                   wfs.kd.ibzk_kc[ik]):
                        # Compare untransformed projections
                        compare_projections(proj, proj_nosym, n, atol)
                        # Compare untransformed wf:s
                        assert np.allclose(abs(ut_nR[n]),
                                           abs(ut_nR_nosym[n]), atol=atol)

                    # Then check so that absolute value of transformed
                    # projections are the same
                    compare_projections(proj_sym, proj_nosym, n, atol)

                    # Check so that periodic part of pseudo is same,
                    # up to a phase
                    assert np.allclose(abs(ut_nR_sym[n]),
                                       abs(ut_nR_nosym[n]), atol=atol)

                # For degenerate states check transformation
                # matrix is unitary,
                # For non-degenerate states check so that all-electron wf:s
                # are the same up to phase
                bands = range(n, n + dim)

                check_all_electron_wfs(bands, ut_nR_sym,
                                       ut_nR_nosym,
                                       proj_sym, proj_nosym, dO_aii,
                                       wfs.gd.dv, atol)
                n += dim


def get_overlap(bands, ut1_nR, ut2_nR, proj1, proj2, dO_aii, dv):
    """ Computes overlap of all-electron wavefunctions
    Similar to gpaw.berryphase.get_overlap but adapted
    to work with projector objects rather than arrays.
    XXX Eventually berryphase.get_overlap should be replaced
    by this function

    Parameters
    ----------
    bands:  integer list
            bands to calculate overlap for
    ut1_nR:  np.array
            ut_nR array
    ut2_nR:  np.array
            ut_nR array
    proj1: GPAW Projections object
    proj2: GPAW Projections object
    dO_aii: dict
            overlaps from setups
    dv:     float
            calc.wfs.gd.dv
    """
    NR = np.prod(np.shape(ut1_nR)[1:])
    ut1_nR = np.reshape(ut1_nR, (len(ut1_nR), NR))
    ut2_nR = np.reshape(ut2_nR, (len(ut2_nR), NR))
    M_nn = (ut1_nR[bands].conj() @ ut2_nR[bands].T) * dv

    for a in proj1.map:
        P1_ni = proj1[a][bands]
        P2_ni = proj2[a][bands]
        dO_ii = dO_aii[a]
        M_nn += P1_ni.conj() @ (dO_ii) @ (P2_ni.T)

    return M_nn


def equal_dicts(dict_1, dict_2, atol):
    """ Checks so that two dicts with np.arrays are
    equal"""
    assert len(dict_1.keys()) == len(dict_2.keys())
    for key in dict_1:
        # Make sure the dictionaries contain the same set of keys
        if key not in dict_2:
            return False
        # Make sure that the arrays are identical
        if not np.allclose(dict_1[key], dict_2[key], atol=atol):
            return False
    return True


def find_degenerate_subspace(eps_n, n_start, nbands, atol_eig):
    # Find degenerate eigenvalues
    n = n_start
    dim = 1
    while n < nbands - 1 and abs(eps_n[n] - eps_n[n + 1]) < atol_eig:
        dim += 1
        n += 1
    return dim


def compare_projections(proj_sym, proj_nosym, n, atol):
    # compares so that projections at given k and band index n
    # differ by at most a phase
    for a, P_ni in proj_sym.items():
        for j in range(P_ni.shape[1]):
            # Check so that absolute values of projections are the same
            assert np.isclose(abs(P_ni[n, j]),
                              abs(proj_nosym[a][n, j]),
                              atol=atol)


def get_overlaps_from_setups(wfs):
    dO_aii = {}
    for a in wfs.kpt_u[0].projections.map:
        dO_aii[a] = wfs.setups[a].dO_ii
    return dO_aii


def check_all_electron_wfs(bands, ut1_nR, ut2_nR,
                           proj_sym, proj_nosym, dO_aii,
                           dv, atol):
    """sets up transformation matrix between symmetry
       transformed u:s and normal u:s in degenerate subspace
       and asserts that it is unitary. It also checks that
       the pseudo wf:s transform according to the same
       transformation.

       Let |ψ^1_i> denote the all electron wavefunctions
       from the calculation with symmetry and |ψ^2_i>
       the corresponding wavefunctions from the calculation
       without symmetry.
       If the set {|ψ^1_i>} span the same subspace as the set
       {|ψ^2_i>} they fulfill the following where summation
       over repeated indexes is assumed:

       |ψ^2_i> = |ψ^1_k> <ψ^1_k |ψ^2_i> == M_ki |ψ^1_k>
       and M_ki = <ψ^1_k |ψ^2_i>  is a unitary transformation.
       M_ki is only unitary if the two sets of wfs span the
       same subspace.

    Parameters
    ---------
    bands: list of ints
         band indexes in degenerate subspace
    ut1_nR: np.array
    ut2_nR: np.array
        Periodic part of pseudo wave function for two calculations
    proj_sym: Projections object
    proj_nosym: Projections object
        Projections for two calculations
    dO_aii: dict with np.arrays
       see get_overlaps_from_setups
    dv:     float
            calc.wfs.gd.dv
    atol: float
       absolute tolerance when comparing arrays
    """
    M_nn = get_overlap(bands,
                       ut1_nR,
                       ut2_nR,
                       proj_sym,
                       proj_nosym,
                       dO_aii,
                       dv)

    # Check so that transformation matrix is unitary
    MMdag_nn = M_nn @ M_nn.T.conj()
    assert np.allclose(np.eye(len(MMdag_nn)), MMdag_nn, atol=atol)

    # Check so that M_nn transforms pseudo wf:s, see docs
    ut2_from_transform_nR = np.einsum('ji,jklm->iklm', M_nn, ut1_nR[bands])
    assert np.allclose(ut2_from_transform_nR, ut2_nR[bands], atol=atol)


def get_ibz_data_from_wfs(wfs, nbands, ik, s):
    """ gets data at ibz k-point ik
    """
    # get energies and wfs
    kpt = wfs.kpt_qs[ik][s]
    psit_nG = kpt.psit_nG
    eps_n = kpt.eps_n

    # Get periodic part of pseudo wfs
    ut_nR = np.array([wfs.pd.ifft(
        psit_nG[n], ik) for n in range(nbands)])

    # Get projections
    proj = kpt.projections.new(nbands=nbands, bcomm=None)
    proj.array[:] = kpt.projections.array[:nbands]

    # get overlaps
    dO_aii = get_overlaps_from_setups(wfs)
    return eps_n, ut_nR, proj, dO_aii
