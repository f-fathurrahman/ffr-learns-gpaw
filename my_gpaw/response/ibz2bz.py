from collections.abc import Sequence
import numpy as np
from my_gpaw.utilities.blas import gemmdot


class IBZ2BZMaps(Sequence):
    """Sequence of data maps from k-points in the IBZ to the full BZ.

    The sequence is indexed by the BZ index K."""

    def __init__(self, kd, spos_ac, R_asii, N_c):
        """Construct the IBZ2BZMapper.

        Parameters
        ----------
        kd : KPointDescriptor
        spos_ac : np.array
            Scaled atomic positions
        R_asii : list
            Rotations of the PAW projections under symmetry transformations
        N_c : iterable of 3 ints
            Number of grid points along axes
        """
        self.kd = kd
        self.spos_ac = spos_ac
        self.R_asii = R_asii

        # Scaled coordinates of the real-space grid
        self.r_cR = np.array(np.meshgrid(*[np.linspace(0, 1, N, endpoint=False)
                                           for N in N_c], indexing='ij'))

    @classmethod
    def from_calculator(cls, calc):
        R_asii = [setup.R_sii for setup in calc.setups]
        return cls(calc.wfs.kd, calc.spos_ac, R_asii, calc.wfs.gd.N_c)

    def __len__(self):
        return len(self.kd.bzk_kc)

    def __getitem__(self, K):
        kd = self.kd
        return IBZ2BZMap(kd.ibzk_kc[kd.bz2ibz_k[K]],
                         *self.get_symmetry_transformations(kd.sym_k[K]),
                         kd.time_reversal_k[K],
                         self.spos_ac,
                         self.kd.bzk_kc[K],
                         self.r_cR)

    def get_symmetry_transformations(self, s):
        return (self.get_rotation_matrix(s),
                self.get_atomic_permutations(s),
                self.get_projections_rotations(s))

    def get_rotation_matrix(self, s):
        """Coordinate rotation matrix, mapping IBZ -> K."""
        U_cc = self.kd.symmetry.op_scc[s]
        return U_cc

    def get_atomic_permutations(self, s):
        """Permutations of atomic indices in the IBZ -> K map."""
        b_a = self.kd.symmetry.a_sa[s]  # Atom a is mapped onto atom b
        return b_a

    def get_projections_rotations(self, s):
        """Rotations of the PAW projections for the IBZ -> K mapping."""
        R_aii = [R_sii[s] for R_sii in self.R_asii]
        return R_aii


class IBZ2BZMap:
    """Functionality to map orbitals from the IBZ to a specific k-point K."""

    def __init__(self, ik_c, U_cc, b_a, R_aii,
                 time_reversal, spos_ac, k_c, r_cR):
        """Construct the IBZ2BZMap."""
        self.ik_c = ik_c
        self.k_c = k_c  # k_c in 1:st BZ that IBZ-kpoint is mapped to
        self.r_cR = r_cR
        self.U_cc = U_cc
        self.b_a = b_a
        self.R_aii = R_aii
        self.time_reversal = time_reversal

        self.spos_ac = spos_ac

    def map_kpoint(self):
        """Get the relative k-point coordinates after the IBZ -> K mapping.

        NB: The mapped k-point can lie outside the BZ, but will always be
        related to kd.bzk_kc[K] by a reciprocal lattice vector.
        """
        # Apply symmetry operations to the irreducible k-point
        sign = 1 - 2 * self.time_reversal
        k_c = sign * self.U_cc @ self.ik_c

        return k_c

    def map_pseudo_wave(self, ut_R):
        """Map the periodic part of the pseudo wave from the IBZ -> K.

        For the symmetry operator U, which maps K = U ik, where ik is the
        irreducible BZ k-point and K does not necessarily lie within the 1BZ,

        psit_K(r) = psit_ik(U^T r)

        The mapping takes place on the coarse real-space grid.
        """
        # Apply symmetry operations to the periodic part of the pseudo wave
        if not (self.U_cc == np.eye(3)).all():
            N_c = ut_R.shape
            i_cr = np.dot(self.U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            utout_R = ut_R.ravel()[i].reshape(N_c)
        else:
            utout_R = ut_R.copy()
        if self.time_reversal:
            utout_R = utout_R.conj()

        assert utout_R is not ut_R,\
            "We don't want the output array to point back at the input array"

        return utout_R

    def map_pseudo_wave_to_BZ(self, ut_R):
        """Map the periodic part of wave function from IBZ -> k in BZ.

        Parameters
        ----------
        ut_R: np.array
              Periodic part of pseudo wf at IBZ k-point
        """
        # Map the pseudo wave and K-point using symmetry operations
        utout_R = self.map_pseudo_wave(ut_R)
        kpt_shift_c = self.map_kpoint() - self.k_c

        # Check if the resulting K-point already resides inside the BZ
        if np.allclose(kpt_shift_c, 0.0):
            return utout_R

        # Check that the mapped k-point differ from the BZ K-point by
        # a reciprocal lattice vector G
        assert np.allclose((kpt_shift_c - np.round(kpt_shift_c)), 0.0)

        # Add a phase e^iG.r to the periodic part of the wf
        return utout_R * np.exp(2j * np.pi * gemmdot(kpt_shift_c, self.r_cR))

    def map_projections(self, projections):
        """Perform IBZ -> K mapping of the PAW projections.

        The projections of atom "a" are mapped onto an atom related to atom
        "b" by a lattice vector:

        r_b = U^T r_a    (or equivalently: r_b^T = r_a^T U)

        This means that when mapping

        psi_K(r) = psi_ik(U^T r),

        we need to generate the projections at atom a for k-point K based on
        the projections at atom b for k-point ik.
        """
        mapped_projections = projections.new()
        for a, (b, U_ii) in enumerate(zip(self.b_a, self.U_aii)):
            # Map projections
            Pin_ni = projections[b]
            Pout_ni = Pin_ni @ U_ii
            if self.time_reversal:
                Pout_ni = np.conj(Pout_ni)

            # Store output projections
            I1, I2 = mapped_projections.map[a]
            mapped_projections.array[..., I1:I2] = Pout_ni

        return mapped_projections
        
    @property
    def U_aii(self):
        """Phase corrected rotation matrices for the PAW projections."""
        U_aii = []
        for a, R_ii in enumerate(self.R_aii):
            # The symmetry transformation maps atom "a" to a position which is
            # related to atom "b" by a lattice vector (but which does not
            # necessarily lie within the unit cell)
            b = self.b_a[a]
            cell_shift_c = self.spos_ac[a] @ self.U_cc - self.spos_ac[b]
            assert np.allclose(cell_shift_c.round(), cell_shift_c)
            # This means, that when we want to extract the projections at K for
            # atom a according to psi_K(r_a) = psi_ik(U^T r_a), we need the
            # projections at U^T r_a for k-point ik. Since we only have the
            # projections within the unit cell we need to multiply them with a
            # phase factor according to the cell shift.
            phase_factor = np.exp(2j * np.pi * self.ik_c @ cell_shift_c)
            U_ii = R_ii.T * phase_factor
            U_aii.append(U_ii)

        return U_aii
