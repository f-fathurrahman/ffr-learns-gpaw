"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

from itertools import product, combinations

import numpy as np
import pytest
from my_gpaw import GPAW
from my_gpaw.mpi import world
from my_gpaw.response import ResponseContext, ResponseGroundStateAdapter
from my_gpaw.response.frequencies import ComplexFrequencyDescriptor
from my_gpaw.response.chiks import ChiKSCalculator
from my_gpaw.response.chi0 import Chi0
from my_gpaw.response.pair_functions import (get_inverted_pw_mapping,
                                          get_pw_coordinates)
from my_gpaw.test.conftest import response_band_cutoff

# ---------- chiks parametrization ---------- #


def generate_system_s(spincomponents=['00', '+-']):
    # Compute chiks for different materials and spin components
    system_s = [  # wfs, spincomponent
        ('fancy_si_pw_wfs', '00'),
        ('al_pw_wfs', '00'),
        ('fe_pw_wfs', '00'),
        ('fe_pw_wfs', '+-'),
        ('co_pw_wfs', '00'),
        ('co_pw_wfs', '+-'),
    ]

    # Filter spincomponents
    system_s = [system for system in system_s if system[1] in spincomponents]

    return system_s


def generate_qrel_q():
    # Fractional q-vectors on a path towards a reciprocal lattice vector
    qrel_q = np.array([0., 0.25, 0.5])

    return qrel_q


def get_q_c(wfs, qrel):
    if wfs in ['fancy_si_pw_wfs', 'al_pw_wfs']:
        # Generate points on the G-X path
        q_c = qrel * np.array([1., 0., 1.])
    elif wfs == 'fe_pw_wfs':
        # Generate points on the G-N path
        q_c = qrel * np.array([0., 0., 1.])
    elif wfs == 'co_pw_wfs':
        # Generate points on the G-M path
        q_c = qrel * np.array([1., 0., 0.])
    else:
        raise ValueError('Invalid wfs', wfs)

    return q_c


def get_tolerances(system, qrel):
    # Define tolerance for each test system
    wfs, spincomponent = system
    identifier = wfs + '_' + spincomponent

    # Si and Fe the density-density response has perfect symmetry
    atols = {
        'fancy_si_pw_wfs_00': 1e-8,
        'fe_pw_wfs_00': 1e-8,
    }

    # For Al, the symmetries are not perfectly conserved, but worst for the
    # q-point q_X
    if qrel == 0.0:
        al_atol = 1e-6
    elif qrel == 0.25:
        al_atol = 5e-5
    elif qrel == 0.5:
        al_atol = 2e-4
    atols['al_pw_wfs_00'] = al_atol

    # For Fe, the symmetries are not perfectly conserved for the
    # transverse magnetic response
    if qrel == 0.0:
        fet_atol = 2e-3
    elif qrel == 0.25:
        fet_atol = 8e-3
    elif qrel == 0.5:
        fet_atol = 5e-4
    atols['fe_pw_wfs_+-'] = fet_atol

    # For the density-density reponse in Co, the symmetries are not perfectly
    # conserved for any of the q-points, but quite well conserved for q = 0
    if qrel == 0.0:
        co_atol = 5e-5
    elif qrel == 0.25:
        co_atol = 5e-3
    elif qrel == 0.5:
        co_atol = 1e-3
    atols['co_pw_wfs_00'] = co_atol

    # For the transverse magnetic response in Co, the symmetries are not
    # perfectly conserved for any of the q-points, but again quite well
    # conserved for q = 0
    if qrel == 0.0:
        cot_atol = 5e-5
    elif qrel == 0.25:
        cot_atol = 5e-4
    elif qrel == 0.5:
        cot_atol = 5e-4
    atols['co_pw_wfs_+-'] = cot_atol

    if identifier not in atols.keys():
        raise ValueError(system, qrel)

    atol = atols[identifier]
    rtol = 1e-5

    return atol, rtol


def generate_gc_g():
    # Compute chiks both on a gamma-centered and a q-centered pw grid
    gc_g = [True, False]

    return gc_g


def generate_nblocks_n():
    nblocks_n = [1]
    if world.size % 2 == 0:
        nblocks_n.append(2)
    if world.size % 4 == 0:
        nblocks_n.append(4)

    return nblocks_n


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.parametrize(
    'system,qrel,gammacentered',
    product(generate_system_s(), generate_qrel_q(), generate_gc_g()))
def test_chiks(in_tmp_dir, gpw_files, system, qrel, gammacentered):
    r"""Test the internals of the ChiKSCalculator.

    In particular, we test that the susceptibility does not change due to the
    details in the internal calculator, such as varrying block distribution,
    band summation scheme, reducing the k-point integral using symmetries or
    basing the ground state adapter on a dynamic (and distributed) GPAW
    calculator.

    Furthermore, we test the symmetries of the calculated susceptibilities.
    """
    # ---------- Inputs ---------- #

    # Part 1: Set up ChiKSTestingFactory
    wfs, spincomponent = system
    atol, rtol = get_tolerances(system, qrel)
    q_c = get_q_c(wfs, qrel)

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    complex_frequencies = list(frequencies + 0.j) + list(frequencies + 0.1j)
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    # Part 2: Check toggling of calculation parameters
    # Note: None of these should change the actual results.
    disable_syms_s = [True, False]

    nblocks_n = generate_nblocks_n()
    nn = len(nblocks_n)

    bandsummation_b = ['double', 'pairwise']
    distribution_d = ['GZg', 'ZgG']

    # Symmetry independent tolerances (relating to chiks distribution)
    dist_atol = 1e-8
    dist_rtol = 1e-6

    # Part 3: Check reciprocity and inversion symmetry

    # ---------- Script ---------- #

    # Part 1: Set up ChiKSTestingFactory
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = response_band_cutoff[wfs]

    chiks_testing_factory = ChiKSTestingFactory(calc,
                                                spincomponent, q_c, zd,
                                                nbands, ecut, gammacentered)

    # Part 2: Check toggling of calculation parameters

    # Check symmetry toggle and cross-tabulate with nblocks and bandsummation
    chiks_testing_factory.check_parameter_self_consistency(
        parameter='disable_syms', values=disable_syms_s,
        atol=atol, rtol=rtol,
        cross_tabulation=dict(nblocks=nblocks_n,
                              bandsummation=bandsummation_b))

    # Check nblocks and cross-tabulate with disable_syms and bandsummation
    for n1, n2 in combinations(range(nn), 2):
        chiks_testing_factory.check_parameter_self_consistency(
            parameter='nblocks', values=[nblocks_n[n1], nblocks_n[n2]],
            atol=dist_atol, rtol=dist_rtol,
            cross_tabulation=dict(disable_syms=disable_syms_s,
                                  bandsummation=bandsummation_b))

    # Check bandsummation and cross-tabulate with disable_syms and nblocks
    chiks_testing_factory.check_parameter_self_consistency(
        parameter='bandsummation', values=bandsummation_b,
        atol=atol, rtol=rtol,
        cross_tabulation=dict(disable_syms=disable_syms_s,
                              nblocks=nblocks_n))

    # Check internal distribution and cross-tabulate with nblocks
    chiks_testing_factory.check_parameter_self_consistency(
        parameter='distribution', values=distribution_d,
        atol=dist_atol, rtol=dist_rtol,
        cross_tabulation=dict(nblocks=nblocks_n))

    # Part 3: Check reciprocity and inversion symmetry

    # Cross-tabulate disable_syms, nblocks and bandsummation
    chiks_testing_factory.check_reciprocity_and_inversion_symmetry(
        atol=atol, rtol=rtol,
        cross_tabulation=dict(disable_syms=disable_syms_s,
                              nblocks=nblocks_n,
                              bandsummation=bandsummation_b))

    # Cross-tabulate distribution and nblocks
    chiks_testing_factory.check_reciprocity_and_inversion_symmetry(
        atol=atol, rtol=rtol,
        cross_tabulation=dict(distribution=distribution_d,
                              nblocks=nblocks_n))


@pytest.mark.response
@pytest.mark.parametrize(
    'system,qrel',
    product(generate_system_s(spincomponents=['00']), generate_qrel_q()))
def test_chiks_vs_chi0(in_tmp_dir, gpw_files, system, qrel):
    """Test that the ChiKSCalculator is able to reproduce the Chi0Body.

    We use only the default calculation parameter setup for the ChiKSCalculator
    and leave parameter cross-validation to the test above."""

    # ---------- Inputs ---------- #

    # Part 1: chiks calculation
    wfs, spincomponent = system
    atol, rtol = get_tolerances(system, qrel)
    q_c = get_q_c(wfs, qrel)

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    eta = 0.15
    complex_frequencies = frequencies + 1.j * eta

    # Part 2: chi0 calculation

    # Part 3: Check chiks vs. chi0

    # ---------- Script ---------- #

    # Part 1: chiks calculation

    # Initialize ground state adapter
    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[wfs], context)
    nbands = response_band_cutoff[wfs]

    # Set up complex frequency descriptor
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    # Calculate chiks
    chiks_calc = ChiKSCalculator(gs, context=context,
                                 ecut=ecut, nbands=nbands)
    chiks = chiks_calc.calculate(spincomponent, q_c, zd)
    chiks = chiks.copy_with_global_frequency_distribution()

    # Part 2: chi0 calculation
    chi0_calc = Chi0(gpw_files[wfs],
                     frequencies=frequencies, eta=eta,
                     ecut=ecut, nbands=nbands,
                     hilbert=False, intraband=False)
    chi0_data = chi0_calc.calculate(q_c)
    chi0_wGG = chi0_data.get_distributed_frequencies_array()

    # Part 3: Check chiks vs. chi0
    assert chiks.array == pytest.approx(chi0_wGG, rel=rtol, abs=atol)


# ---------- Test functionality ---------- #


class ChiKSTestingFactory:
    """Factory to calculate and cache chiks objects."""

    def __init__(self, calc,
                 spincomponent, q_c, zd,
                 nbands, ecut, gammacentered):
        self.gs = GSAdapterWithPAWCache(calc)
        self.spincomponent = spincomponent
        self.q_c = q_c
        self.zd = zd
        self.nbands = nbands
        self.ecut = ecut
        self.gammacentered = gammacentered

        self.cached_chiks = {}

    def __call__(self,
                 qsign: int = 1,
                 distribution: str = 'GZg',
                 disable_syms: bool = False,
                 bandsummation: str = 'pairwise',
                 nblocks: int = 1):
        # Compile a string of the calculation parameters for cache look-up
        cache_string = f'{qsign},{distribution},{disable_syms}'
        cache_string += f',{bandsummation},{nblocks}'

        if cache_string in self.cached_chiks:
            return self.cached_chiks[cache_string]

        chiks_calc = ChiKSCalculator(
            self.gs, ecut=self.ecut, nbands=self.nbands,
            gammacentered=self.gammacentered,
            disable_time_reversal=disable_syms,
            disable_point_group=disable_syms,
            bandsummation=bandsummation,
            nblocks=nblocks)

        # Do a manual calculation of chiks
        chiks = chiks_calc._calculate(*chiks_calc._set_up_internals(
            self.spincomponent, qsign * self.q_c, self.zd,
            distribution=distribution))

        chiks = chiks.copy_with_global_frequency_distribution()
        self.cached_chiks[cache_string] = chiks

        return chiks

    def check_parameter_self_consistency(self,
                                         parameter: str, values: list,
                                         atol: float,
                                         rtol: float,
                                         cross_tabulation: dict):
        assert len(values) == 2
        for kwargs in self.generate_cross_tabulated_kwargs(cross_tabulation):
            kwargs[parameter] = values[0]
            chiks1 = self(**kwargs)
            kwargs[parameter] = values[1]
            chiks2 = self(**kwargs)
            compare_pw_bases(chiks1, chiks2)
            assert chiks2.array == pytest.approx(
                chiks1.array, rel=rtol, abs=atol), f'{kwargs}'

    def check_reciprocity_and_inversion_symmetry(self,
                                                 atol: float,
                                                 rtol: float,
                                                 cross_tabulation: dict):
        for kwargs in self.generate_cross_tabulated_kwargs(cross_tabulation):
            # Calculate chiks in q and -q
            chiks1 = self(**kwargs)
            if np.allclose(self.q_c, 0.):
                chiks2 = chiks1
            else:
                chiks2 = self(qsign=-1, **kwargs)
            check_reciprocity_and_inversion_symmetry(chiks1, chiks2,
                                                     atol=atol, rtol=rtol)

    @staticmethod
    def generate_cross_tabulated_kwargs(cross_tabulation: dict):
        # Set up cross tabulation of calculation parameters
        cross_tabulator = product(*[[(key, value)
                                     for value in cross_tabulation[key]]
                                    for key in cross_tabulation])
        for cross_tabulated_parameters in cross_tabulator:
            yield {key: value for key, value in cross_tabulated_parameters}


class GSAdapterWithPAWCache(ResponseGroundStateAdapter):
    """Add a PAW correction cache to the ground state adapter.

    WARNING: Use with care! The cache is only valid, when the plane-wave
    representations are identical.
    """

    def __init__(self, calc):
        super().__init__(calc)

        self._cached_corrections = []
        self._cached_parameters = []

    def pair_density_paw_corrections(self, qpd):
        """Overwrite method with a cached version."""
        cache_index = self._cache_lookup(qpd)
        if cache_index is not None:
            return self._cached_corrections[cache_index]

        return self._calculate_correction(qpd)

    def _calculate_correction(self, qpd):
        correction = super().pair_density_paw_corrections(qpd)

        self._cached_corrections.append(correction)
        self._cached_parameters.append((qpd.q_c, qpd.ecut, qpd.gammacentered))

        return correction

    def _cache_lookup(self, qpd):
        for i, (q_c, ecut,
                gammacentered) in enumerate(self._cached_parameters):
            if np.allclose(qpd.q_c, q_c) and abs(qpd.ecut - ecut) < 1e-8\
               and qpd.gammacentered == gammacentered:
                # Cache hit!
                return i


def compare_pw_bases(chiks1, chiks2):
    """Compare the plane-wave representations of two calculated chiks."""
    G1_Gc = get_pw_coordinates(chiks1.qpd)
    G2_Gc = get_pw_coordinates(chiks2.qpd)
    assert G1_Gc.shape == G2_Gc.shape
    assert np.allclose(G1_Gc - G2_Gc, 0.)


def check_reciprocity_and_inversion_symmetry(chiks1, chiks2, *, atol, rtol):
    """Check the susceptibilities for reciprocity and inversion symmetry

    In particular, we test the reciprocity relation (valid both for μν=00 and
    μν=+-),

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G'-G)^(μν)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G-G')^(μν)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,G'G)^(μν)(q, ω),

    for a real life periodic systems with an inversion center.

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries cannot be fulfilled exactly. Generally speaking,
    the "symmetry" noise can be reduced by running with symmetry='off' in
    the ground state calculation.
    """
    invmap_GG = get_inverted_pw_mapping(chiks1.qpd, chiks2.qpd)

    # Loop over frequencies
    for chi1_GG, chi2_GG in zip(chiks1.array, chiks2.array):
        # Check the reciprocity
        assert chi2_GG[invmap_GG].T == pytest.approx(chi1_GG,
                                                     rel=rtol, abs=atol)
        # Check inversion symmetry
        assert chi2_GG[invmap_GG] == pytest.approx(chi1_GG, rel=rtol, abs=atol)

    # Loop over q-vectors
    for chiks in [chiks1, chiks2]:
        for chiks_GG in chiks.array:  # array = chiks_zGG
            # Check that the full susceptibility matrix is symmetric
            assert chiks_GG.T == pytest.approx(chiks_GG, rel=rtol, abs=atol)
