"""Module for evaluating two-center integrals.

Contains classes for evaluating integrals of the form::

             /
            |   _   _a    _   _b   _
    Theta = | f(r - R ) g(r - R ) dr ,
            |
           /

with f and g each being given as a radial function times a spherical
harmonic.

Important classes
-----------------

Low-level:

 * OverlapExpansion: evaluate the overlap between a pair of functions (or a
   function with itself) for some displacement vector: <f | g>.  An overlap
   expansion can be created once for a pair of splines f and g, and actual
   values of the overlap can then be evaluated for several different
   displacement vectors.
 * FourierTransformer: create OverlapExpansion object from pair of splines.

Mid-level:

 * TwoSiteOverlapExpansions: evaluate overlaps between two *sets* of functions,
   where functions in the same set reside on the same location: <f_j | g_j>.
 * TwoSiteOverlapCalculator: create TwoSiteOverlapExpansions object from
   pair of lists of splines.

High-level:

 * ManySiteOverlapExpansions:  evaluate overlaps with many functions in many
   locations: <f_aj | g_aj>.
 * ManySiteOverlapCalculator: create ManySiteOverlapExpansions object from
   pair of lists of splines nested by atom and orbital number.

The low-level classes do the actual work, while the higher-level ones depend
on the lower-level ones.

"""

from math import pi

import numpy as np

from ase.neighborlist import PrimitiveNeighborList
from ase.data import covalent_radii
from ase.units import Bohr

import my_gpaw25.cgpaw as cgpaw
from my_gpaw25.ffbt import ffbt, FourierBesselTransformer
from my_gpaw25.gaunt import gaunt
from my_gpaw25.spherical_harmonics import Yl, nablarlYL
from my_gpaw25.spline import Spline
from my_gpaw25.utilities.tools import tri2full
from my_gpaw25.utilities.timing import nulltimer

timer = nulltimer  # XXX global timer object, only for hacking

UL = 'L'


class BaseOverlapExpansionSet:
    def __init__(self, shape):
        self.shape = shape

    def zeros(self, shape=(), dtype=float):
        return np.zeros(shape + self.shape, dtype=dtype)


class OverlapExpansion(BaseOverlapExpansionSet):
    """A list of real-space splines representing an overlap integral."""
    def __init__(self, la, lb, spline_l):
        self.la = la
        self.lb = lb
        self.lmaxgaunt = max(la, lb)
        self.spline_l = spline_l
        self.lmaxspline = (la + lb) % 2 + 2 * len(self.spline_l)
        BaseOverlapExpansionSet.__init__(self, (2 * la + 1, 2 * lb + 1))
        self.cspline_l = [spline.spline for spline in self.spline_l]

    def evaluate(self, r, rlY_lm, G_LLL, x_mi, _nil=np.empty(0)):
        cgpaw.tci_overlap(self.la, self.lb, G_LLL, self.cspline_l,
                          r, rlY_lm, x_mi,
                          False, _nil, _nil, _nil)

    def derivative(self, r, Rhat_c, rlY_L, G_LLL, drlYdR_Lc, dxdR_cmi,
                   _nil=np.empty(0)):
        # timer.start('deriv')
        cgpaw.tci_overlap(self.la, self.lb, G_LLL, self.cspline_l,
                          r, rlY_L, _nil,
                          True, Rhat_c, drlYdR_Lc, dxdR_cmi)
        # timer.stop('deriv')


class TwoSiteOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, la_j, lb_j, oe_jj):
        self.oe_jj = oe_jj
        shape = (sum([2 * l + 1 for l in la_j]),
                 sum([2 * l + 1 for l in lb_j]))
        BaseOverlapExpansionSet.__init__(self, shape)
        if oe_jj.size == 0:
            self.lmaxgaunt = 0
            self.lmaxspline = 0
        else:
            self.lmaxgaunt = max(oe.lmaxgaunt for oe in oe_jj.ravel())
            self.lmaxspline = max(oe.lmaxspline for oe in oe_jj.ravel())

    def slice(self, x_xMM):
        assert x_xMM.shape[-2:] == self.shape
        Ma1 = 0
        for j1, oe_j in enumerate(self.oe_jj):
            Mb1 = 0
            Ma2 = Ma1
            for j2, oe in enumerate(oe_j):
                Ma2 = Ma1 + oe.shape[0]
                Mb2 = Mb1 + oe.shape[1]
                yield x_xMM[..., Ma1:Ma2, Mb1:Mb2], oe
                Mb1 = Mb2
            Ma1 = Ma2

    def evaluate(self, r, rlY_lm):
        timer.start('tsoe eval')
        x_MM = self.zeros()
        G_LLL = gaunt(self.lmaxgaunt)
        rlY_L = rlY_lm.toarray(self.lmaxspline)
        for x_mm, oe in self.slice(x_MM):
            oe.evaluate(r, rlY_L, G_LLL, x_mm)
        timer.stop('tsoe eval')
        return x_MM

    def derivative(self, r, Rhat, rlY_lm, drlYdR_lmc):
        x_cMM = self.zeros((3,))
        G_LLL = gaunt(self.lmaxgaunt)
        rlY_L = rlY_lm.toarray(self.lmaxspline)
        drlYdR_Lc = drlYdR_lmc.toarray(self.lmaxspline)
        # print(drlYdR_lmc.shape)
        for x_cmm, oe in self.slice(x_cMM):
            oe.derivative(r, Rhat, rlY_L, G_LLL, drlYdR_Lc, x_cmm)
        return x_cMM


class ManySiteOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, tsoe_II, I1_a, I2_a):
        self.tsoe_II = tsoe_II
        self.I1_a = I1_a
        self.I2_a = I2_a

        M1 = 0
        M1_a = []
        for I in I1_a:
            M1_a.append(M1)
            M1 += tsoe_II[I, 0].shape[0]
        self.M1_a = M1_a

        M2 = 0
        M2_a = []
        for I in I2_a:
            M2_a.append(M2)
            M2 += tsoe_II[0, I].shape[1]
        self.M2_a = M2_a

        shape = (sum([tsoe_II[I, 0].shape[0] for I in I1_a]),
                 sum([tsoe_II[0, I].shape[1] for I in I2_a]))
        assert (M1, M2) == shape
        BaseOverlapExpansionSet.__init__(self, shape)

    def get(self, a1, a2):
        return self.tsoe_II[self.I1_a[a1], self.I2_a[a2]]

    def getslice(self, a1, a2, x_xMM):
        I1 = self.I1_a[a1]
        I2 = self.I2_a[a2]
        tsoe = self.tsoe_II[I1, I2]
        Mstart1 = self.M1_a[a1]
        Mend1 = Mstart1 + tsoe.shape[0]
        Mstart2 = self.M2_a[a2]
        Mend2 = Mstart2 + tsoe.shape[1]
        return x_xMM[..., Mstart1:Mend1, Mstart2:Mend2], tsoe

    def evaluate_slice(self, disp, x_qxMM):
        x_qxmm, oe = self.getslice(disp.a1, disp.a2, x_qxMM)
        disp.evaluate_overlap(oe, x_qxmm)


class DomainDecomposedExpansions(BaseOverlapExpansionSet):
    def __init__(self, msoe, local_indices):
        self.msoe = msoe
        self.local_indices = local_indices
        BaseOverlapExpansionSet.__init__(self, msoe.shape)

    def evaluate_slice(self, disp, x_xqMM):
        if disp.a2 in self.local_indices:
            self.msoe.evaluate_slice(disp, x_xqMM)


class ManySiteDictionaryWrapper(DomainDecomposedExpansions):
    # Used with dictionaries such as P_aqMi and dPdR_aqcMi

    def getslice(self, a1, a2, xdict_aqxMi):
        msoe = self.msoe
        tsoe = msoe.tsoe_II[msoe.I1_a[a1], msoe.I2_a[a2]]
        Mstart = self.msoe.M1_a[a1]
        Mend = Mstart + tsoe.shape[0]
        return xdict_aqxMi[a2][..., Mstart:Mend, :], tsoe

    def evaluate_slice(self, disp, x_aqxMi):
        if disp.a2 in x_aqxMi:
            x_qxmi, oe = self.getslice(disp.a1, disp.a2, x_aqxMi)
            disp.evaluate_overlap(oe, x_qxmi)
        if disp.a1 in x_aqxMi and (disp.a1 != disp.a2):
            x2_qxmi, oe2 = self.getslice(disp.a2, disp.a1, x_aqxMi)
            rdisp = disp.reverse()
            rdisp.evaluate_overlap(oe2, x2_qxmi)


class BlacsOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, msoe, local_indices, Mmystart, mynao):
        self.msoe = msoe
        self.local_indices = local_indices
        BaseOverlapExpansionSet.__init__(self, msoe.shape)

        self.Mmystart = Mmystart
        self.mynao = mynao

        M_a = msoe.M1_a
        natoms = len(M_a)
        a = 0
        while a < natoms and M_a[a] <= Mmystart:
            a += 1
        a -= 1
        self.astart = a

        while a < natoms and M_a[a] < Mmystart + mynao:
            a += 1
        self.aend = a

    def evaluate_slice(self, disp, x_xqNM):
        a1 = disp.a1
        a2 = disp.a2
        if a2 in self.local_indices and (self.astart <= a1 < self.aend):
            # assert a2 <= a1
            msoe = self.msoe
            I1 = msoe.I1_a[a1]
            I2 = msoe.I2_a[a2]
            tsoe = msoe.tsoe_II[I1, I2]
            x_qxmm = tsoe.zeros(x_xqNM.shape[:-2], dtype=x_xqNM.dtype)
            disp.evaluate_overlap(tsoe, x_qxmm)
            Mstart1 = msoe.M1_a[a1] - self.Mmystart
            Mend1 = Mstart1 + tsoe.shape[0]
            Mstart1b = max(0, Mstart1)
            Mend1b = min(self.mynao, Mend1)
            Mstart2 = msoe.M2_a[a2]
            Mend2 = Mstart2 + tsoe.shape[1]
            x_xqNM[..., Mstart1b:Mend1b, Mstart2:Mend2] += \
                x_qxmm[..., Mstart1b - Mstart1:Mend1b - Mstart1, :]
        # This is all right as long as we are only interested in a2 <= a1
        # if a1 in self.local_indices and a2 < a1 and (self.astart <=
        #                                              a2 < self.aend):
        #     self.evaluate_slice(disp.reverse(), x_xqNM)


class NeighborPairs:
    """Class for looping over pairs of atoms using a neighbor list."""
    def __init__(self, cutoff_a, cell_cv, pbc_c, self_interaction):
        self.neighbors = PrimitiveNeighborList(
            cutoff_a, skin=0, sorted=True,
            self_interaction=self_interaction,
            use_scaled_positions=True)
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.neighbors.update(self.pbc_c, self.cell_cv, spos_ac)

    def iter(self):
        cell_cv = self.cell_cv
        for a1, spos1_c in enumerate(self.spos_ac):
            a2_a, offsets = self.neighbors.get_neighbors(a1)
            for a2, offset in zip(a2_a, offsets):
                spos2_c = self.spos_ac[a2] + offset
                R_c = np.dot(spos2_c - spos1_c, cell_cv)
                yield a1, a2, R_c, offset


class PairFilter:
    def __init__(self, pairs):
        self.pairs = pairs

    def set_positions(self, spos_ac):
        self.pairs.set_positions(spos_ac)

    def iter(self):
        return self.pairs.iter()


class PairsWithSelfinteraction(PairFilter):
    def iter(self):
        for a1, a2, R_c, offset in self.pairs.iter():
            yield a1, a2, R_c, offset
            if a1 == a2 and offset.any():
                yield a1, a1, -R_c, -offset


class OppositeDirection(PairFilter):
    def iter(self):
        for a1, a2, R_c, offset in self.pairs.iter():
            yield a2, a1, -R_c, -offset


class FourierTransformer(FourierBesselTransformer):
    def calculate_overlap_expansion(self, phit1phit2_q, l1, l2):
        """Calculate list of splines for one overlap integral.

        Given two Fourier transformed functions, return list of splines
        in real space necessary to evaluate their overlap.

          phi  (q) * phi  (q) --> [phi    (r), ..., phi    (r)] .
             l1         l2            lmin             lmax

        The overlap <phi1 | phi2> can then be calculated by linear
        combinations of the returned splines with appropriate Gaunt
        coefficients.
        """
        lmax = l1 + l2
        splines = []
        R = np.arange(self.Q) * self.dr
        R1 = R.copy()
        R1[0] = 1.0
        k1 = self.k_q.copy()
        k1[0] = 1.0
        a_q = phit1phit2_q
        for l in range(lmax % 2, lmax + 1, 2):
            a_g = (8 * ffbt(l, a_q * k1**(-2 - lmax - l), self.k_q, R) /
                   R1**(2 * l + 1))
            if l == 0:
                a_g[0] = 8 * np.sum(a_q * k1**(-lmax)) * self.dk
            else:
                a_g[0] = a_g[1]  # XXXX
            a_g *= (-1)**((l1 - l2 - l) // 2)
            n = len(a_g) // 256
            s = Spline.from_data(
                l, 2 * self.rcut, np.concatenate((a_g[::n], [0.0])),
            )
            splines.append(s)
        return OverlapExpansion(l1, l2, splines)

    def laplacian(self, f_jq):
        return 0.5 * f_jq * self.k_q**2.0


class TwoSiteOverlapCalculator:
    def __init__(self, transformer):
        self.transformer = transformer

    def transform(self, f_j):
        f_jq = np.array([self.transformer.transform(f) for f in f_j])
        return f_jq

    def calculate_expansions(self, la_j, fa_jq, lb_j, fb_jq):
        nja = len(la_j)
        njb = len(lb_j)
        assert nja == len(fa_jq) and njb == len(fb_jq)
        oe_jj = np.empty((nja, njb), dtype=object)
        for ja, (la, fa_q) in enumerate(zip(la_j, fa_jq)):
            for jb, (lb, fb_q) in enumerate(zip(lb_j, fb_jq)):
                a_q = fa_q * fb_q
                oe = self.transformer.calculate_overlap_expansion(a_q, la, lb)
                oe_jj[ja, jb] = oe
        return TwoSiteOverlapExpansions(la_j, lb_j, oe_jj)

    def calculate_kinetic_expansions(self, l_j, f_jq):
        t_jq = self.transformer.laplacian(f_jq)
        return self.calculate_expansions(l_j, f_jq, l_j, t_jq)

    def laplacian(self, f_jq):
        t_jq = self.transformer.laplacian(f_jq)
        return t_jq


class ManySiteOverlapCalculator:
    def __init__(self, twosite_calculator, I1_a, I2_a):
        """Create VeryManyOverlaps object.

        twosite_calculator: instance of TwoSiteOverlapCalculator
        I_a: mapping from atom index (as in spos_a) to unique atom type"""
        self.twosite_calculator = twosite_calculator
        self.I1_a = I1_a
        self.I2_a = I2_a

    def transform(self, f_Ij):
        f_Ijq = [self.twosite_calculator.transform(f_j) for f_j in f_Ij]
        return f_Ijq

    def calculate_expansions(self, l1_Ij, f1_Ijq, l2_Ij, f2_Ijq):
        # Naive implementation, just loop over everything
        # We should only need half of them
        nI1 = len(l1_Ij)
        nI2 = len(l2_Ij)
        assert len(l1_Ij) == len(f1_Ijq) and len(l2_Ij) == len(f2_Ijq)
        tsoe_II = np.empty((nI1, nI2), dtype=object)
        calc = self.twosite_calculator
        for I1, (l1_j, f1_jq) in enumerate(zip(l1_Ij, f1_Ijq)):
            for I2, (l2_j, f2_jq) in enumerate(zip(l2_Ij, f2_Ijq)):
                tsoe = calc.calculate_expansions(l1_j, f1_jq, l2_j, f2_jq)
                tsoe_II[I1, I2] = tsoe
        return ManySiteOverlapExpansions(tsoe_II, self.I1_a, self.I2_a)

    def calculate_kinetic_expansions(self, l_Ij, f_Ijq):
        t_Ijq = [self.twosite_calculator.laplacian(f_jq) for f_jq in f_Ijq]
        return self.calculate_expansions(l_Ij, f_Ijq, l_Ij, t_Ijq)


class AtomicDisplacement:
    def __init__(self, factory, a1, a2, R_c, offset, phases):
        self.factory = factory
        self.a1 = a1
        self.a2 = a2
        self.R_c = R_c
        self.offset = offset
        self.phases = phases
        self.r = np.linalg.norm(R_c)
        self._set_spherical_harmonics(R_c)

    def _set_spherical_harmonics(self, R_c):
        self.rlY_lm = LazySphericalHarmonics(R_c)

    # XXX new
    def evaluate_direct(self, oe, dst_xqmm):
        src_xmm = self.evaluate_direct_without_phases(oe)
        self.phases.apply(src_xmm, dst_xqmm)

    # XXX new
    def evaluate_direct_without_phases(self, oe):
        return oe.evaluate(self.r, self.rlY_lm)

    # XXX clean up unnecessary methods
    def overlap_without_phases(self, oe):
        return oe.evaluate(self.r, self.rlY_lm)

    def _evaluate_without_phases(self, oe):
        return self.overlap_without_phases(oe)

    def evaluate_overlap(self, oe, dst_xqmm):
        src_xmm = self._evaluate_without_phases(oe)
        timer.start('phases')
        self.phases.apply(src_xmm, dst_xqmm)
        timer.stop('phases')

    def reverse(self):
        return self.factory.displacementclass(self.factory, self.a2, self.a1,
                                              -self.R_c, -self.offset,
                                              self.phases.inverse())


class LazySphericalHarmonics:
    """Class for caching spherical harmonics as they are calculated.

    Behaves like a list Y_lm, but really calculates (or retrieves) Y_m
    once a given value of l is __getitem__'d."""
    def __init__(self, R_c):
        self.R_c = np.asarray(R_c)
        self.Y_lm = {}
        self.Y_lm1 = np.empty(0)
        # self.dYdr_lmc = {}
        self.lmax = 0

    def evaluate(self, l):
        rlY_m = np.empty(2 * l + 1)
        Yl(l, self.R_c, rlY_m)
        return rlY_m

    def __getitem__(self, l):
        Y_m = self.Y_lm.get(l)
        if Y_m is None:
            Y_m = self.evaluate(l)
            self.Y_lm[l] = Y_m
        return Y_m

    def toarray(self, lmax):
        if lmax > self.lmax:
            self.Y_lm1 = np.concatenate([self.Y_lm1] +
                                        [self.evaluate(l).ravel()
                                         for l in range(self.lmax, lmax)])
            self.lmax = lmax
        return self.Y_lm1


class LazySphericalHarmonicsDerivative(LazySphericalHarmonics):
    def evaluate(self, l):
        drlYdR_mc = np.empty((2 * l + 1, 3))
        L0 = l**2
        for m in range(2 * l + 1):
            drlYdR_mc[m, :] = nablarlYL(L0 + m, self.R_c)
        return drlYdR_mc


class DerivativeAtomicDisplacement(AtomicDisplacement):
    def _set_spherical_harmonics(self, R_c):
        self.rlY_lm = LazySphericalHarmonics(R_c)
        self.drlYdr_lmc = LazySphericalHarmonicsDerivative(R_c)

        if R_c.any():
            self.Rhat_c = R_c / self.r
        else:
            self.Rhat_c = np.zeros(3)

    def derivative_without_phases(self, oe):
        return oe.derivative(self.r, self.Rhat_c, self.rlY_lm, self.drlYdr_lmc)

    def _evaluate_without_phases(self, oe):  # override
        return self.derivative_without_phases(oe)


class NullPhases:
    def __init__(self, ibzk_qc, offset):
        pass

    def apply(self, src_xMM, dst_qxMM):
        assert len(dst_qxMM) == 1
        dst_qxMM[0][:] += src_xMM

    def inverse(self):
        return self


class BlochPhases:
    def __init__(self, ibzk_qc, offset):
        self.phase_q = np.exp(-2j * pi * np.dot(ibzk_qc, offset))
        self.ibzk_qc = ibzk_qc
        self.offset = offset

    def apply(self, src_xMM, dst_qxMM):
        assert dst_qxMM.dtype == complex, dst_qxMM.dtype
        for phase, dst_xMM in zip(self.phase_q, dst_qxMM):
            dst_xMM[:] += phase * src_xMM

    def inverse(self):
        return BlochPhases(-self.ibzk_qc, self.offset)


class TwoCenterIntegralCalculator:
    # This class knows how to apply phases, and whether to call the
    # various derivative() or evaluate() methods
    def __init__(self, ibzk_qc=None, derivative=False):
        if derivative:
            displacementclass = DerivativeAtomicDisplacement
        else:
            displacementclass = AtomicDisplacement
        self.displacementclass = displacementclass

        if ibzk_qc is None or not ibzk_qc.any():
            self.phaseclass = NullPhases
        else:
            self.phaseclass = BlochPhases
        self.ibzk_qc = ibzk_qc
        self.derivative = derivative

    def calculate(self, atompairs, expansions, arrays):
        for disp in self.iter(atompairs):
            for expansion, X_qxMM in zip(expansions, arrays):
                expansion.evaluate_slice(disp, X_qxMM)

    def iter(self, atompairs):
        for a1, a2, R_c, offset in atompairs.iter():
            # if a1 == a2 and self.derivative:
            #     continue
            phase_applier = self.phaseclass(self.ibzk_qc, offset)
            yield self.displacementclass(self, a1, a2, R_c, offset,
                                         phase_applier)


class NewTwoCenterIntegrals:
    def __init__(self, cell_cv, pbc_c, setups, ibzk_qc, gamma):
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.ibzk_qc = ibzk_qc
        self.gamma = gamma

        timer.start('tci init')
        cutoff_I = []
        setups_I = setups.setups.values()
        I_setup = {}
        for I, setup in enumerate(setups_I):
            I_setup[setup] = I
            cutoff_I.append(max([func.get_cutoff()
                                 for func
                                 in setup.basis_functions_J + setup.pt_j]))

        I_a = []
        for setup in setups:
            I_a.append(I_setup[setup])

        cutoff_a = [cutoff_I[I] for I in I_a]

        self.cutoff_a = cutoff_a  # convenient for writing the new new overlap
        self.I_a = I_a
        self.setups_I = setups_I
        self.atompairs = PairsWithSelfinteraction(NeighborPairs(cutoff_a,
                                                                cell_cv,
                                                                pbc_c,
                                                                True))

        scale = 0.01  # XXX minimal distance scale
        cutoff_close_a = [covalent_radii[int(s.Z)] / Bohr * scale
                          for s in setups]
        self.atoms_close = NeighborPairs(cutoff_close_a, cell_cv, pbc_c, False)

        rcut = max(cutoff_I + [0.001])
        transformer = FourierTransformer(rcut, N=2**10)
        tsoc = TwoSiteOverlapCalculator(transformer)
        self.msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)
        self.calculate_expansions()

        self.calculate = self.evaluate  # XXX compatibility

        self.set_matrix_distribution(None, None)
        timer.stop('tci init')

    def set_matrix_distribution(self, Mmystart, mynao):
        """Distribute matrices using BLACS."""
        # Range of basis functions for BLACS distribution of matrices:
        self.Mmystart = Mmystart
        self.mynao = mynao
        self.blacs = mynao is not None

    def calculate_expansions(self):
        timer.start('tci calc exp')
        phit_Ij = [setup.basis_functions_J for setup in self.setups_I]
        l_Ij = []
        for phit_j in phit_Ij:
            l_Ij.append([phit.get_angular_momentum_number()
                         for phit in phit_j])

        pt_l_Ij = [setup.l_j for setup in self.setups_I]
        pt_Ij = [setup.pt_j for setup in self.setups_I]
        phit_Ijq = self.msoc.transform(phit_Ij)
        pt_Ijq = self.msoc.transform(pt_Ij)

        msoc = self.msoc

        self.Theta_expansions = msoc.calculate_expansions(l_Ij, phit_Ijq,
                                                          l_Ij, phit_Ijq)
        self.T_expansions = msoc.calculate_kinetic_expansions(l_Ij, phit_Ijq)
        self.P_expansions = msoc.calculate_expansions(l_Ij, phit_Ijq,
                                                      pt_l_Ij, pt_Ijq)
        timer.stop('tci calc exp')

    def _calculate(self, calc, spos_ac, Theta_qxMM, T_qxMM, P_aqxMi):
        Theta_qxMM.fill(0.0)
        T_qxMM.fill(0.0)
        for P_qxMi in P_aqxMi.values():
            P_qxMi.fill(0.0)

        if 1:  # XXX
            self.atoms_close.set_positions(spos_ac)
            wrk = [x for x in self.atoms_close.iter()]
            if len(wrk) != 0:
                txt = ''
                for a1, a2, R_c, offset in wrk:
                    txt += 'Atom %d and Atom %d in cell (%d, %d, %d)\n' % \
                        (a1, a2, offset[0], offset[1], offset[2])
                raise RuntimeError('Atoms too close!\n' + txt)

        self.atompairs.set_positions(spos_ac)

        if self.blacs:
            # S and T matrices are distributed:
            expansions = [
                BlacsOverlapExpansions(self.Theta_expansions,
                                       P_aqxMi, self.Mmystart, self.mynao),
                BlacsOverlapExpansions(self.T_expansions,
                                       P_aqxMi, self.Mmystart, self.mynao)]
        else:
            expansions = [DomainDecomposedExpansions(self.Theta_expansions,
                                                     P_aqxMi),
                          DomainDecomposedExpansions(self.T_expansions,
                                                     P_aqxMi)]

        expansions.append(ManySiteDictionaryWrapper(self.P_expansions,
                                                    P_aqxMi))
        arrays = [Theta_qxMM, T_qxMM, P_aqxMi]
        timer.start('tci calculate')
        calc.calculate(OppositeDirection(self.atompairs), expansions, arrays)
        timer.stop('tci calculate')

    def evaluate(self, spos_ac, Theta_qMM, T_qMM, P_aqMi):
        calc = TwoCenterIntegralCalculator(self.ibzk_qc, derivative=False)
        self._calculate(calc, spos_ac, Theta_qMM, T_qMM, P_aqMi)
        if not self.blacs:
            for X_MM in list(Theta_qMM) + list(T_qMM):
                tri2full(X_MM, UL=UL)

    def derivative(self, spos_ac, dThetadR_qcMM, dTdR_qcMM, dPdR_aqcMi):
        calc = TwoCenterIntegralCalculator(self.ibzk_qc, derivative=True)
        self._calculate(calc, spos_ac, dThetadR_qcMM, dTdR_qcMM, dPdR_aqcMi)

        def antihermitian(src, dst):
            np.conj(-src, dst)

        if not self.blacs:
            for X_cMM in list(dThetadR_qcMM) + list(dTdR_qcMM):
                for X_MM in X_cMM:
                    tri2full(X_MM, UL=UL, map=antihermitian)

    calculate_derivative = derivative  # XXX compatibility

    def estimate_memory(self, mem):
        mem.subnode('TCI not impl.', 0)
