import numpy as np
from time import ctime

from ase.units import Hartree

from my_gpaw.utilities.blas import mmmx

from my_gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from my_gpaw.response.frequencies import ComplexFrequencyDescriptor
from my_gpaw.response.pw_parallelization import PlaneWaveBlockDistributor
from my_gpaw.response.matrix_elements import NewPairDensityCalculator
from my_gpaw.response.pair_integrator import PairFunctionIntegrator
from my_gpaw.response.pair_transitions import PairTransitions
from my_gpaw.response.pair_functions import SingleQPWDescriptor, Chi


class ChiKSCalculator(PairFunctionIntegrator):
    r"""Calculator class for the four-component Kohn-Sham susceptibility tensor
    of collinear systems in absence of spin-orbit coupling,
    see [PRB 103, 245110 (2021)]:
                              __  __   __
                           1  \   \    \
    χ_KS,GG'^μν(q,ω+iη) =  ‾  /   /    /   σ^μ_ss' σ^ν_s's (f_nks - f_n'k+qs')
                           V  ‾‾  ‾‾   ‾‾
                              k   n,n' s,s'
                                        n_nks,n'k+qs'(G+q) n_n'k+qs',nks(-G'-q)
                                      x ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                            ħω - (ε_n'k+qs' - ε_nks) + iħη

    where the matrix elements

    n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

    are the unit cell normalized plane-wave pair densities of each transition.
    """

    def __init__(self, gs: ResponseGroundStateAdapter, context=None,
                 nblocks=1,
                 ecut=50, gammacentered=False,
                 nbands=None,
                 bandsummation='pairwise',
                 **kwargs):
        """Contruct the ChiKSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nblocks : int
            Distribute the chiks_zGG array into nblocks (where nblocks is a
            divisor of context.comm.size)
        ecut : float (or None)
            Plane-wave cutoff in eV
        gammacentered : bool
            Center the grid of plane waves around the Γ-point (or the q-vector)
        nbands : int
            Number of bands to include in the sum over states
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        kwargs : see gpaw.response.pair_integrator.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, nblocks=nblocks, **kwargs)

        self.ecut = None if ecut is None else ecut / Hartree  # eV to Hartree
        self.gammacentered = gammacentered
        self.nbands = nbands
        self.bandsummation = bandsummation

        self.pair_density_calc = NewPairDensityCalculator(gs, context)

    def calculate(self, spincomponent, q_c, zd) -> Chi:
        r"""Calculate χ_KS,GG'^μν(q,z), where z = ω + iη

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the Kohn-Sham susceptibility.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or np.array
            Wave vector in relative coordinates
        zd : ComplexFrequencyDescriptor
            Complex frequencies z to evaluate χ_KS,GG'^μν(q,z) at.
        """
        return self._calculate(*self._set_up_internals(spincomponent, q_c, zd))

    def _set_up_internals(self, spincomponent, q_c, zd,
                          distribution='GZg'):
        r"""Set up internal data objects to calculate χ_KS."""
        assert isinstance(zd, ComplexFrequencyDescriptor)

        # Set up the internal plane-wave descriptor
        qpdi = self.get_pw_descriptor(q_c, internal=True)

        # Prepare to sum over bands and spins
        transitions = self.get_band_and_spin_transitions(
            spincomponent, nbands=self.nbands,
            bandsummation=self.bandsummation)

        self.context.print(self.get_info_string(
            qpdi, len(zd), spincomponent, len(transitions)))

        # Create data structure
        chiks = self.create_chiks(spincomponent, qpdi, zd, distribution)

        return chiks, transitions

    def _calculate(self, chiks: Chi, transitions: PairTransitions):
        r"""Integrate χ_KS according to the specified transitions."""
        self.context.print('Initializing pair density PAW corrections')
        self.pair_density_calc.initialize_paw_corrections(chiks.qpd)

        # Perform the actual integration
        analyzer = self._integrate(chiks, transitions)

        # Symmetrize chiks according to the symmetries of the ground state
        self.symmetrize(chiks, analyzer)

        # Map to standard output format
        chiks = self.post_process(chiks)

        return chiks

    def get_pw_descriptor(self, q_c, internal=False):
        """Get plane-wave descriptor for the wave vector q_c.

        Parameters
        ----------
        q_c : list or ndarray
            Wave vector in relative coordinates
        internal : bool
            When using symmetries, the actual calculation of chiks must happen
            using a q-centered plane wave basis. If internal==True, as it is by
            default, the internal plane wave basis (used in the integration of
            chiks.array) is returned, otherwise the external descriptor is
            returned, corresponding to the requested chiks.
        """
        q_c = np.asarray(q_c, dtype=float)
        gd = self.gs.gd

        # Update to internal basis, if needed
        if internal and self.gammacentered and not self.disable_symmetries:
            # In order to make use of the symmetries of the system to reduce
            # the k-point integration, the internal code assumes a plane wave
            # basis which is centered at q in reciprocal space.
            gammacentered = False
            # If we want to compute the pair function on a plane wave grid
            # which is effectively centered in the gamma point instead of q, we
            # need to extend the internal ecut such that the q-centered grid
            # encompasses all reciprocal lattice points inside the gamma-
            # centered sphere.
            # The reduction to the global gamma-centered basis will then be
            # carried out as a post processing step.

            # Compute the extended internal ecut
            B_cv = 2.0 * np.pi * gd.icell_cv  # Reciprocal lattice vectors
            q_v = q_c @ B_cv
            ecut = get_ecut_to_encompass_centered_sphere(q_v, self.ecut)
        else:
            gammacentered = self.gammacentered
            ecut = self.ecut

        qpd = SingleQPWDescriptor.from_q(q_c, ecut, gd,
                                         gammacentered=gammacentered)

        return qpd

    def create_chiks(self, spincomponent, qpd, zd, distribution):
        """Create a new Chi object to be integrated."""
        assert distribution in ['GZg', 'ZgG']
        blockdist = PlaneWaveBlockDistributor(self.context.comm,
                                              self.blockcomm,
                                              self.intrablockcomm)
        return Chi(spincomponent, qpd, zd,
                   blockdist, distribution=distribution)

    @timer('Add integrand to chiks')
    def add_integrand(self, kptpair, weight, chiks):
        r"""Use the NewPairDensityCalculator object to calculate the integrand
        for all relevant transitions of the given k-point pair, k -> k + q.

        Depending on the bandsummation parameter, the integrand of the
        collinear four-component Kohn-Sham susceptibility tensor (in the
        absence of spin-orbit coupling) is calculated as:

        bandsummation: double

                   __
                   \  σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k =  /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ n_kt(G+q) n_kt^*(G'+q)
                   ‾‾      ħz - (ε_n'k's' - ε_nks)
                   t

        where n_kt(G+q) = n_nks,n'k+qs'(G+q) and

        bandsummation: pairwise

                    __ /
                    \  | σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k =   /  | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                    ‾‾ |      ħz - (ε_n'k's' - ε_nks)
                    t  \
                                                       \
                    σ^μ_s's σ^ν_ss' (f_nks - f_n'k's') |
           - δ_n'>n ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ | n_kt(G+q) n_kt^*(G'+q)
                         ħz + (ε_n'k's' - ε_nks)       |
                                                       /

        The integrand is added to the output array chiks_x multiplied with the
        supplied kptpair integral weight.
        """
        # Calculate the pair densities n_kt(G+q)
        pair_density = self.pair_density_calc(kptpair, chiks.qpd)

        # Extract the temporal ingredients from the KohnShamKPointPair
        transitions = kptpair.transitions  # transition indices (n,s)->(n',s')
        df_t = kptpair.df_t  # (f_n'k's' - f_nks)
        deps_t = kptpair.deps_t  # (ε_n'k's' - ε_nks)

        # Calculate the temporal part of the integrand
        if chiks.spincomponent == '00' and self.gs.nspins == 1:
            weight = 2 * weight
        x_Zt = get_temporal_part(chiks.spincomponent, chiks.zd.hz_z,
                                 transitions, df_t, deps_t,
                                 self.bandsummation)

        self._add_integrand(pair_density, x_Zt, weight, chiks)

    def _add_integrand(self, pair_density, x_Zt, weight, chiks):
        r"""Add the integrand to chiks.

        This entail performing a sum of transition t and an outer product
        in the pair density plane wave components G and G',
                    __
                    \
        (...)_k =   /  x_t^μν(ħz) n_kt(G+q) n_kt^*(G'+q)
                    ‾‾
                    t

        where x_t^μν(ħz) is the temporal part of χ_KS,GG'^μν(q,ω+iη).
        """
        if chiks.distribution == 'ZgG':
            self._add_integrand_ZgG(pair_density, x_Zt, weight, chiks)
        elif chiks.distribution == 'GZg':
            self._add_integrand_GZg(pair_density, x_Zt, weight, chiks)
        else:
            raise ValueError(f'Invalid distribution {chiks.distribution}')

    def _add_integrand_ZgG(self, pair_density, x_Zt, weight, chiks):
        """Add integrand in ZgG distribution.

        Z = global complex frequency index
        g = distributed G plane wave index
        G = global G' plane wave index
        """
        chiks_ZgG = chiks.array
        myslice = chiks.blocks1d.myslice

        with self.context.timer('Set up ncc and xn'):
            # Multiply the temporal part with the k-point integration weight
            x_Zt *= weight

            # Set up n_kt^*(G'+q)
            n_tG = pair_density.get_global_array()
            ncc_tG = n_tG.conj()

            # Set up x_t^μν(ħz) n_kt(G+q)
            n_gt = np.ascontiguousarray(n_tG[:, myslice].T)
            xn_Zgt = x_Zt[:, np.newaxis, :] * n_gt[np.newaxis, :, :]

        with self.context.timer('Perform sum over t-transitions of xn * ncc'):
            for xn_gt, chiks_gG in zip(xn_Zgt, chiks_ZgG):
                mmmx(1.0, xn_gt, 'N', ncc_tG, 'N', 1.0, chiks_gG)  # slow step

    def _add_integrand_GZg(self, pair_density, x_Zt, weight, chiks):
        """Add integrand in GZg distribution.

        G = global G' plane wave index
        Z = global complex frequency index
        g = distributed G plane wave index
        """
        chiks_GZg = chiks.array
        myslice = chiks.blocks1d.myslice

        with self.context.timer('Set up ncc and xn'):
            # Multiply the temporal part with the k-point integration weight
            x_tZ = np.ascontiguousarray(weight * x_Zt.T)

            # Set up n_kt^*(G'+q)
            n_tG = pair_density.get_global_array()
            n_Gt = np.ascontiguousarray(n_tG.T)
            ncc_Gt = n_Gt.conj()

            # Set up x_t^μν(ħz) n_kt(G+q)
            n_tg = n_tG[:, myslice]
            xn_tZg = x_tZ[:, :, np.newaxis] * n_tg[:, np.newaxis, :]

        with self.context.timer('Perform sum over t-transitions of ncc * xn'):
            mmmx(1.0, ncc_Gt, 'N', xn_tZg, 'N', 1.0, chiks_GZg)  # slow step

    @timer('Symmetrizing chiks')
    def symmetrize(self, chiks, analyzer):
        """Symmetrize chiks_zGG."""
        chiks_ZgG = chiks.array_with_view('ZgG')

        # Distribute over frequencies
        nz = len(chiks.zd)
        tmp_zGG = chiks.blockdist.distribute_as(chiks_ZgG, nz, 'zGG')
        analyzer.symmetrize_zGG(tmp_zGG)
        # Distribute over plane waves
        chiks_ZgG[:] = chiks.blockdist.distribute_as(tmp_zGG, nz, 'ZgG')

    @timer('Post processing')
    def post_process(self, chiks):
        """Cast a calculated chiks into a fixed output format."""
        if chiks.distribution != 'ZgG':
            # Always output chiks with distribution 'ZgG'
            chiks = chiks.copy_with_distribution('ZgG')

        if self.gammacentered and not self.disable_symmetries:
            # Reduce the q-centered plane-wave basis used internally to the
            # gammacentered basis
            assert not chiks.qpd.gammacentered  # Internal qpd
            qpd = self.get_pw_descriptor(chiks.q_c)  # External qpd
            chiks = chiks.copy_with_reduced_pd(qpd)

        return chiks

    def get_info_string(self, qpd, nz, spincomponent, nt):
        r"""Get information about the χ_KS,GG'^μν(q,z) calculation"""
        from my_gpaw.utilities.memory import maxrss

        q_c = qpd.q_c
        ecut = qpd.ecut * Hartree
        Asize = nz * qpd.ngmax**2 * 16. / 1024**2 / self.blockcomm.size
        cmem = maxrss() / 1024**2

        s = '\n'

        s += 'Setting up a Kohn-Sham susceptibility calculation with:\n'
        s += '    Spin component: %s\n' % spincomponent
        s += '    q_c: [%f, %f, %f]\n' % (q_c[0], q_c[1], q_c[2])
        s += '    Number of frequency points: %d\n' % nz
        if self.nbands is None:
            s += '    Bands included: All\n'
        else:
            s += '    Number of bands included: %d\n' % self.nbands
        s += 'Resulting in:\n'
        s += '    A total number of band and spin transitions of: %d\n' % nt
        s += '\n'

        s += self.get_basic_info_string()
        s += '\n'

        s += 'Plane-wave basis of the Kohn-Sham susceptibility:\n'
        s += '    Planewave cutoff: %f\n' % ecut
        s += '    Number of planewaves: %d\n' % qpd.ngmax
        s += '    Memory estimates:\n'
        s += '        A_zGG: %f M / cpu\n' % Asize
        s += '        Memory usage before allocation: %f M / cpu\n' % cmem
        s += '\n'
        s += '%s\n' % ctime()

        return s


def get_ecut_to_encompass_centered_sphere(q_v, ecut):
    """Calculate the minimal ecut which results in a q-centered plane wave
    basis containing all the reciprocal lattice vectors G, which lie inside a
    specific gamma-centered sphere:

    |G|^2 < 2 * ecut
    """
    q = np.linalg.norm(q_v)
    ecut += q * (np.sqrt(2 * ecut) + q / 2)

    return ecut


def get_temporal_part(spincomponent, hz_z,
                      transitions, df_t, deps_t,
                      bandsummation):
    """Get the temporal part of a (causal linear) susceptibility, x_t^μν(ħz).
    """
    _get_temporal_part = create_get_temporal_part(bandsummation)
    return _get_temporal_part(spincomponent, hz_z,
                              transitions, df_t, deps_t)


def create_get_temporal_part(bandsummation):
    """Creator component, deciding how to calculate the temporal part"""
    if bandsummation == 'double':
        return get_double_temporal_part
    elif bandsummation == 'pairwise':
        return get_pairwise_temporal_part
    raise ValueError(bandsummation)


def get_double_temporal_part(spincomponent, hz_z,
                             transitions, df_t, deps_t):
    r"""Get:

                 σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
    x_t^μν(ħz) = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                      ħz - (ε_n'k's' - ε_nks)
    """
    # Get the right spin components
    s1_t, s2_t = transitions.get_spin_indices()
    scomps_t = get_smat_components(spincomponent, s1_t, s2_t)
    # Calculate nominator
    nom_t = - scomps_t * df_t  # df = f2 - f1
    # Calculate denominator
    denom_wt = hz_z[:, np.newaxis] - deps_t[np.newaxis, :]  # de = e2 - e1

    regularize_intraband_transitions(denom_wt, transitions, deps_t)

    return nom_t[np.newaxis, :] / denom_wt


def get_pairwise_temporal_part(spincomponent, hz_z,
                               transitions, df_t, deps_t):
    r"""Get:

                 /
                 | σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
    x_t^μν(ħz) = | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                 |      ħz - (ε_n'k's' - ε_nks)
                 \
                                                               \
                            σ^μ_s's σ^ν_ss' (f_nks - f_n'k's') |
                   - δ_n'>n ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ |
                                 ħz + (ε_n'k's' - ε_nks)       |
                                                               /
    """
    n1_t, n2_t, s1_t, s2_t = transitions.get_band_and_spin_indices()
    # Kroenecker delta
    delta_t = np.ones(len(n1_t))
    delta_t[n2_t <= n1_t] = 0
    # Get the right spin components
    scomps1_t = get_smat_components(spincomponent, s1_t, s2_t)
    scomps2_t = get_smat_components(spincomponent, s2_t, s1_t)
    # Calculate nominators
    nom1_t = - scomps1_t * df_t  # df = f2 - f1
    nom2_t = - delta_t * scomps2_t * df_t
    # Calculate denominators
    denom1_wt = hz_z[:, np.newaxis] - deps_t[np.newaxis, :]  # de = e2 - e1
    denom2_wt = hz_z[:, np.newaxis] + deps_t[np.newaxis, :]

    regularize_intraband_transitions(denom1_wt, transitions, deps_t)
    regularize_intraband_transitions(denom2_wt, transitions, deps_t)

    return nom1_t[np.newaxis, :] / denom1_wt\
        - nom2_t[np.newaxis, :] / denom2_wt


def regularize_intraband_transitions(denom_wt, transitions, deps_t):
    """Regularize the denominator of the temporal part in case of degeneracy.

    If the q-vector connects two symmetrically equivalent k-points inside a
    band, the occupation differences vanish and we regularize the denominator.

    NB: In principle there *should* be a contribution from the intraband
    transitions, but this is left for future work for now."""
    intraband_t = transitions.get_intraband_mask()
    degenerate_t = np.abs(deps_t) < 1e-8

    denom_wt[:, intraband_t & degenerate_t] = 1.


def get_smat_components(spincomponent, s1_t, s2_t):
    """For s1=s and s2=s', get:
    smu_ss' snu_s's
    """
    smatmu = smat(spincomponent[0])
    smatnu = smat(spincomponent[1])

    return smatmu[s1_t, s2_t] * smatnu[s2_t, s1_t]


def smat(spinrot):
    if spinrot == '0':
        return np.array([[1, 0], [0, 1]])
    elif spinrot == 'u':
        return np.array([[1, 0], [0, 0]])
    elif spinrot == 'd':
        return np.array([[0, 0], [0, 1]])
    elif spinrot == '-':
        return np.array([[0, 0], [1, 0]])
    elif spinrot == '+':
        return np.array([[0, 1], [0, 0]])
    else:
        raise ValueError(spinrot)
