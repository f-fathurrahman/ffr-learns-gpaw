from time import ctime
from functools import partial

import numpy as np
from ase.units import Ha

from my_gpaw.response.integrators import Integrand
from my_gpaw.response.chi0 import Chi0Calculator
from my_gpaw.response.pair_functions import SingleQPWDescriptor
from my_gpaw.response.chi0_data import Chi0DrudeData
from my_gpaw.response.frequencies import FrequencyGridDescriptor


class Chi0DrudeCalculator(Chi0Calculator):
    """Class for calculating the plasma frequency contribution to Chi0,
    that is, the contribution from intraband transitions inside of metallic
    bands. This corresponds directly to the dielectric function in the Drude
    model."""

    def __init__(self, pair,
                 disable_point_group=False,
                 disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 integrationmode=None):
        self.pair = pair
        self.gs = pair.gs
        self.context = pair.context

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode

        # Number of completely filled bands and number of non-empty bands.
        self.nocc1, self.nocc2 = self.gs.count_occupied_bands()

    def calculate(self, wd, rate, spin='all'):
        """Calculate the Drude dielectric response.

        Parameters
        ----------
        wd : FrequencyDescriptor
            Frequencies to evaluate the reponse function at.
        rate : float
            Plasma frequency decay rate (in eV), corresponding to the
            imaginary part of the complex frequency.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.
        """
        self.print_info(wd, rate)

        # Parse the spin input
        spins = self.get_spins(spin)

        chi0_drude = Chi0DrudeData.from_frequency_descriptor(wd, rate)
        self._calculate(chi0_drude, spins)

        return chi0_drude

    def _calculate(self, chi0_drude: Chi0DrudeData, spins):
        """In-place calculation of the Drude dielectric response function,
        based on the free-space plasma frequency of the intraband transitions.
        """
        # Create a dummy plane-wave descriptor. We need this for the symmetry
        # analysis -> see discussion in gpaw.response.jdos
        qpd = SingleQPWDescriptor.from_q([0., 0., 0.],
                                         ecut=1e-3, gd=self.gs.gd)

        integrator = self.initialize_integrator()
        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        kind, extraargs = self.get_integral_kind()

        integrand = PlasmaFrequencyIntegrand(self, qpd, analyzer)

        # Integrate using temporary array
        tmp_plasmafreq_wvv = np.zeros((1,) + chi0_drude.vv_shape, complex)
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=integrand,
                             out_wxx=tmp_plasmafreq_wvv,  # Output array
                             **extraargs)  # Extra args for int. method
        tmp_plasmafreq_wvv *= prefactor

        # Store the plasma frequency itself and print it for anyone to use
        plasmafreq_vv = tmp_plasmafreq_wvv[0].copy()
        analyzer.symmetrize_wvv(plasmafreq_vv[np.newaxis])
        chi0_drude.plasmafreq_vv += 4 * np.pi * plasmafreq_vv
        self.context.print('Plasma frequency:', flush=False)
        self.context.print((chi0_drude.plasmafreq_vv**0.5 * Ha).round(2))

        # Calculate the Drude dielectric response function from the
        # free-space plasma frequency
        # χ_D(ω+iη) = ω_p^2 / (ω+iη)^2
        assert chi0_drude.zd.upper_half_plane
        chi0_drude.chi_Zvv += plasmafreq_vv[np.newaxis] \
            / chi0_drude.zd.hz_z[:, np.newaxis, np.newaxis]**2

    def update_integrator_kwargs(self, *unused, **ignored):
        """The Drude calculator uses only standard integrator kwargs."""
        pass

    def get_integral_kind(self):
        """Define what "kind" of integral to make."""
        extraargs = {}
        # The plasma frequency integral is special in the way, that only
        # the spectral part is needed
        kind = 'spectral function'
        if self.integrationmode is None:
            # Calculate intraband transitions at finite fermi smearing
            extraargs['intraband'] = True  # Calculate intraband
        elif self.integrationmode == 'tetrahedron integration':
            # Calculate intraband transitions at T=0
            fermi_level = self.gs.fermi_level
            extraargs['x'] = FrequencyGridDescriptor([-fermi_level])

        return kind, extraargs

    def print_info(self, wd, rate):
        p = partial(self.context.print, flush=False)

        p('%s' % ctime())
        p('Called chi0_drude.calculate() with:')
        p('    Number of frequency points: %d' % len(wd))
        p('    Plasma frequency decay rate: %f eV' % rate)
        p()
        p(self.get_gs_info_string(tab='    '))
        self.context.print('')


class PlasmaFrequencyIntegrand(Integrand):
    def __init__(self, chi0drudecalc, qpd, analyzer):
        self._drude = chi0drudecalc
        self.qpd = qpd
        self.analyzer = analyzer

    def _band_summation(self):
        # Intraband response needs only integrate partially unoccupied bands.
        return self._drude.nocc1, self._drude.nocc2

    def matrix_element(self, k_v, s):
        """NB: In dire need of documentation! XXX."""
        n1, n2 = self._band_summation()
        k_c = np.dot(self.qpd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self._drude.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self._drude.pair.intraband_pair_density(kpt1, n_n)

        if self._drude.integrationmode is None:
            f_n = kpt1.f_n
            width = self._drude.gs.get_occupations_width()
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(self.analyzer.get_kpoint_weight(k_c) /
                             self.analyzer.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    def eigenvalues(self, k_v, s):
        """A function that can return the intraband eigenvalues.

        A method describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        n1, n2 = self._band_summation()
        gs = self._drude.gs
        kd = gs.kd
        k_c = np.dot(self.qpd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self._drude.pair.find_kpoint(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = gs.kpt_qs[ik][s]
        assert gs.kd.comm.size == 1

        return kpt1.eps_n[n1:n2]
