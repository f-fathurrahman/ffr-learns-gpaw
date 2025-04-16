"""Calculate the Heisenberg exchange constants in Fe and Co using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""

# General modules
import pytest
import numpy as np

# Script modules
from my_gpaw import GPAW

from my_gpaw.response import ResponseGroundStateAdapter, ResponseContext
from my_gpaw.response.chiks import ChiKSCalculator
from my_gpaw.response.localft import LocalFTCalculator, LocalPAWFTCalculator
from my_gpaw.response.mft import IsotropicExchangeCalculator
from my_gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from my_gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)
from my_gpaw.test.conftest import response_band_cutoff


@pytest.mark.response
def test_Fe_bcc(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 50
    # Do the high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     ])
    # Define site kernels to test
    # Test a single site of spherical and cylindrical geometries
    rc_pa = np.array([[1.0], [1.5], [2.0]])
    hc_pa = np.array([[1.0], [1.5], [2.0]])
    ez_pav = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['fe_pw_wfs'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw_wfs']
    atoms = calc.atoms

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the Heisenberg exchange calculator
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc = ChiKSCalculator(gs, context,
                                 ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    isoexch_calc = IsotropicExchangeCalculator(chiks_calc, localft_calc)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)

    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    mm = 2.21
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mw_qp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)[:, 0, :]

    # Compare results to test values
    test_J_pq = np.array(
        [[2.1907596825086455, 1.172424411323134, 1.6060583789867644],
         [2.612428039019977, 1.2193926800088601, 1.7635196888465006],
         [6.782367391186284, 0.2993922109834177, 1.9346016211386057],
         [1.5764800860123762, 0.8365204592352894, 1.1648584638500161],
         [2.4230224513213234, 1.2179759558303274, 1.6691805687218078],
         [5.35668502504496, 0.3801778545994659, 1.6948968244858478],
         [2.523580017606111, 1.21779750159267, 1.7637120466695273]])
    test_mw_pq = np.array(
        [[0.0, 0.9215703811633589, 0.5291414511510236],
         [0.0, 1.2606654832679791, 0.7682428508357253],
         [0.0, 5.866945864436984, 4.38711834393455],
         [0.0, 0.6696467210652369, 0.3725082553505521],
         [0.0, 1.0905398149239784, 0.682209848506349],
         [0.0, 4.503626398593207, 3.313835475619106],
         [0.0, 1.181703634401304, 0.6876633221145555]])

    # Exchange constants
    assert J_qp.imag == pytest.approx(0.0)
    assert J_qp.T.real == pytest.approx(test_J_pq, rel=2e-3)

    # Magnon energies
    assert mw_qp.T == pytest.approx(test_mw_pq, rel=2e-3)


@pytest.mark.response
def test_Co_hcp(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 100
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # Unfortunately, the usage of symmetry leads to such extensive repetition
    # of random noise, that one cannot trust individual values of J very well.
    # This is improved when increasing the number of k-points, but the problem
    # never completely vanishes
    J_atol = 5.e-3
    J_rtol = 5.e-2
    # However, derived physical values have an increased error cancellation due
    # to their collective nature.
    mw_rtol = 5.e-3  # relative tolerance of absolute results
    mw_ctol = 5.e-2  # relative tolerance on kernel and eta self-consistency

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['co_pw_wfs'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw_wfs']
    atoms = calc.get_atoms()

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)

    # Set up a site kernel to fill out the entire unit cell
    cell_cv = atoms.get_cell()
    cc_v = np.sum(cell_cv, axis=0) / 2.  # Unit cell center
    ucsitekernels = ParallelepipedicSiteKernels([cc_v], [[cell_cv]])

    # Initialize the exchange calculator with and without symmetry
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc0 = ChiKSCalculator(gs, context,
                                  disable_point_group=True,
                                  disable_time_reversal=True,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalPAWFTCalculator(gs, context)
    isoexch_calc0 = IsotropicExchangeCalculator(chiks_calc0, localft_calc)
    chiks_calc1 = ChiKSCalculator(gs, context,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks_calc1, localft_calc)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qs = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels)
        chiksr_buffer = isoexch_calc0._chiksr
        Juc_qs[q, 0] = isoexch_calc0(q_c, ucsitekernels)[0, 0, 0]
        assert isoexch_calc0._chiksr is chiksr_buffer,\
            'Two subsequent IsotropicExchangeCalculator calls with the same '\
            'q_c, should reuse, not update, the chiks buffer'

        Juc_qs[q, 1] = isoexch_calc1(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mom = atoms.get_magnetic_moment()
    mm_ap = mom / 2.0 * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_qs = calculate_single_site_magnon_energies(Juc_qs, q_qc, mom)

    # Compare results to test values
    # print(J_qabp[..., 1])
    # print(mw_qnp[..., 1])
    # print(mwuc_qs[:, 0])
    test_J_qab = np.array([[[1.23106207 - 0.j, 0.25816335 - 0.j],
                            [0.25816335 + 0.j, 1.23106207 + 0.j]],
                           [[0.88823839 + 0.j, 0.07345416 - 0.04947835j],
                            [0.07345416 + 0.04947835j, 0.88823839 + 0.j]],
                           [[1.09349955 - 0.j, 0.00000010 - 0.01176761j],
                            [0.00000010 + 0.01176761j, 1.09349955 - 0.j]]])
    test_mw_qn = np.array([[0., 0.64793939],
                           [0.64304039, 0.86531921],
                           [0.48182997, 0.51136436]])
    test_mwuc_q = np.array([0., 0.69678659, 0.44825874])

    # Exchange constants
    # err = np.absolute(J_qabp[..., 1] - test_J_qab)
    # is_bad = err > J_atol + J_rtol * np.absolute(test_J_qab)
    # print(is_bad)
    # print(np.absolute(err[is_bad] / np.absolute(test_J_qab[is_bad])))
    assert np.allclose(J_qabp[..., 1], test_J_qab,
                       atol=J_atol, rtol=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qs[0, :], 0.)  # Goldstone
    assert np.allclose(mw_qnp[1:, 0, 1], test_mw_qn[1:, 0], rtol=mw_rtol)
    assert np.allclose(mw_qnp[:, 1, 1], test_mw_qn[:, 1], rtol=mw_rtol)
    assert np.allclose(mwuc_qs[1:, 0], test_mwuc_q[1:], rtol=mw_rtol)

    # Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2],
                       test_mw_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2],
                       test_mw_qn[:, 1, np.newaxis], rtol=mw_ctol)
    # Check that symmetry toggle do not change the magnon energies
    assert np.allclose(mwuc_qs[1:, 0], mwuc_qs[1:, 1], rtol=mw_ctol)
