import numpy as np

from my_gpaw.response import timer
from my_gpaw.response.kspair import KohnShamKPointPair
from my_gpaw.response.pair import phase_shifted_fft_indices


class PairDensity:
    """Data class for transition distributed pair density arrays."""

    def __init__(self, tblocks, n_mytG):
        self.tblocks = tblocks
        self.n_mytG = n_mytG

    @classmethod
    def from_qpd(cls, tblocks, qpd):
        n_mytG = qpd.zeros(tblocks.blocksize)
        return cls(tblocks, n_mytG)

    @property
    def local_array_view(self):
        return self.n_mytG[:self.tblocks.nlocal]

    def get_global_array(self):
        """Get the global (all gathered) pair density array n_tG."""
        n_tG = self.tblocks.all_gather(self.n_mytG)

        return n_tG


class NewPairDensityCalculator:
    r"""Class for calculating pair densities

    n_kt(G+q) = n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

                /
              = | dr e^-i(G+q)r ψ_nks^*(r) ψ_n'k+qs'(r)
                /V0

    for a single k-point pair (k, k + q) in the plane-wave mode."""
    def __init__(self, gs, context):
        self.gs = gs
        self.context = context

        # Save PAW correction for all calls with same q_c
        self._pawcorr = None
        self._currentq_c = None

    def initialize_paw_corrections(self, qpd):
        """Initialize the PAW corrections ahead of the actual calculation."""
        self.get_paw_corrections(qpd)

    def get_paw_corrections(self, qpd):
        """Get PAW corrections correcsponding to a specific q-vector."""
        if self._pawcorr is None \
           or not np.allclose(qpd.q_c - self._currentq_c, 0.):
            with self.context.timer('Initialize PAW corrections'):
                self._pawcorr = self.gs.pair_density_paw_corrections(qpd)
                self._currentq_c = qpd.q_c

        return self._pawcorr

    @timer('Calculate pair density')
    def __call__(self, kptpair: KohnShamKPointPair, qpd) -> PairDensity:
        r"""Calculate the pair density for all transitions t.

        In the PAW method, the all-electron pair density is calculated in
        two contributions, the pseudo pair density and a PAW correction,

        n_kt(G+q) = ñ_kt(G+q) + Δn_kt(G+q),

        see [PRB 103, 245110 (2021)] for details.
        """
        # Initialize a blank pair density object
        pair_density = PairDensity.from_qpd(kptpair.tblocks, qpd)
        n_mytG = pair_density.local_array_view

        self.add_pseudo_pair_density(kptpair, qpd, n_mytG)
        self.add_paw_correction(kptpair, qpd, n_mytG)

        return pair_density

    @timer('Calculate the pseudo pair density')
    def add_pseudo_pair_density(self, kptpair, qpd, n_mytG):
        r"""Add the pseudo pair density to an output array.

        The pseudo pair density is first evaluated on the coarse real-space
        grid and then FFT'ed to reciprocal space:

                    /               ˷          ˷
        ñ_kt(G+q) = | dr e^-i(G+q)r ψ_nks^*(r) ψ_n'k+qs'(r)
                    /V0
                                 ˷          ˷
                  = FFT_G[e^-iqr ψ_nks^*(r) ψ_n'k+qs'(r)]
        """
        ikpt1 = kptpair.ikpt1
        ikpt2 = kptpair.ikpt2

        # Map the k-points from the irreducible part of the BZ to the BZ
        # k-point K (up to a reciprocal lattice vector)
        k1_c = self.gs.ibz2bz[kptpair.K1].map_kpoint()
        k2_c = self.gs.ibz2bz[kptpair.K2].map_kpoint()

        # Fourier transform the periodic part of the pseudo waves to the coarse
        # real-space grid and map them to the BZ k-point K (up to the same
        # reciprocal lattice vector as above)
        ut1_hR = self.get_periodic_pseudo_waves(kptpair.K1, ikpt1)
        ut2_hR = self.get_periodic_pseudo_waves(kptpair.K2, ikpt2)

        # Calculate the pseudo pair density in real space, up to a phase of
        # e^(-i[k+q-k']r).
        # This phase does not necessarily vanish, since k2_c only is required
        # to equal k1_c + qpd.q_c modulo a reciprocal lattice vector.
        ut1cc_mytR = ut1_hR[ikpt1.h_myt].conj()
        nt_mytR = ut1cc_mytR * ut2_hR[ikpt2.h_myt]

        # Get the FFT indices corresponding to the Fourier transform
        #                       ˷          ˷
        # FFT_G[e^(-i[k+q-k']r) u_nks^*(r) u_n'k's'(r)]
        Q_G = phase_shifted_fft_indices(k1_c, k2_c, qpd)

        # Add the desired plane-wave components of the FFT'ed pseudo pair
        # density to the output array
        nlocalt = kptpair.tblocks.nlocal
        assert len(n_mytG) == len(nt_mytR) == nlocalt
        for n_G, n_R in zip(n_mytG, nt_mytR):
            n_G[:] += qpd.fft(n_R, 0, Q_G) * qpd.gd.dv

    @timer('Calculate the pair density PAW corrections')
    def add_paw_correction(self, kptpair, qpd, n_mytG):
        r"""Add the pair-density PAW correction to the output array.

        The correction is calculated as a sum over augmentation spheres a
        and projector indices i and j,
                     __  __
                     \   \   ˷     ˷     ˷    ˷
        Δn_kt(G+q) = /   /  <ψ_nks|p_ai><p_aj|ψ_n'k+qs'> Q_aij(G+q)
                     ‾‾  ‾‾
                     a   i,j

        where the pair-density PAW correction tensor is calculated from the
        smooth and all-electron partial waves:

                     /
        Q_aij(G+q) = | dr e^-i(G+q)r [φ_ai^*(r-R_a) φ_aj(r-R_a)
                     /V0                ˷             ˷
                                      - φ_ai^*(r-R_a) φ_aj(r-R_a)]
        """
        ikpt1 = kptpair.ikpt1
        ikpt2 = kptpair.ikpt2

        # Map the projections from the irreducible part of the BZ to the BZ
        # k-point K
        P1h = self.gs.ibz2bz[kptpair.K1].map_projections(ikpt1.Ph)
        P2h = self.gs.ibz2bz[kptpair.K2].map_projections(ikpt2.Ph)

        # Calculate the actual PAW corrections
        Q_aGii = self.get_paw_corrections(qpd).Q_aGii
        P1 = ikpt1.projectors_in_transition_index(P1h)
        P2 = ikpt2.projectors_in_transition_index(P2h)
        for a, Q_Gii in enumerate(Q_aGii):  # Loop over augmentation spheres
            assert P1.atom_partition.comm.size ==\
                P2.atom_partition.comm.size == 1,\
                'We need access to the projections of all atoms'
            P1_myti = P1[a]
            P2_myti = P2[a]
            # Make outer product of the projectors in the projector index i,j
            P1ccP2_mytii = P1_myti.conj()[..., np.newaxis] \
                * P2_myti[:, np.newaxis]
            # Sum over projector indices and add correction to the output
            n_mytG[:] += np.einsum('tij, Gij -> tG', P1ccP2_mytii, Q_Gii)

    def get_periodic_pseudo_waves(self, K, ikpt):
        """FFT the Kohn-Sham orbitals to real space and map them from the
        irreducible k-point to the k-point in question."""
        ut_hR = self.gs.gd.empty(ikpt.nh, self.gs.dtype)
        for h, psit_G in enumerate(ikpt.psit_hG):
            ut_hR[h] = self.gs.ibz2bz[K].map_pseudo_wave(
                self.gs.global_pd.ifft(psit_G, ikpt.ik))

        return ut_hR
