# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange.
"""

from math import exp, ceil
import numpy as np

from my_gpaw25.atom.configurations import core_states
from my_gpaw25.gaunt import gaunt
from my_gpaw25.lfc import LFC
from my_gpaw25.poisson import PoissonSolver
from my_gpaw25.helmholtz import HelmholtzSolver
from my_gpaw25.transformers import Transformer
from my_gpaw25.utilities import (hartree, pack_density, pack_hermitian,
                            packed_index, unpack_density, unpack_hermitian)
from my_gpaw25.utilities.blas import mmm
from my_gpaw25.utilities.tools import symmetrize
from my_gpaw25.xc import XC
from my_gpaw25.xc.functional import XCFunctional
from my_gpaw25.xc.kernel import XCNull
from my_gpaw25.setup import CachedYukawaInteractions


class HybridXCBase(XCFunctional):
    orbital_dependent = True

    def __init__(self, name, stencil=2, hybrid=None, xc=None, omega=None):
        """Mix standard functionals with exact exchange.

        name: str
            Name of hybrid functional.
        hybrid: float
            Fraction of exact exchange.
        xc: str or XCFunctional object
            Standard DFT functional with scaled down exchange.
        """

        rsf_functionals = {    # Parameters can also be taken from libxc
            'CAMY-BLYP': {  # Akinaga, Ten-no CPL 462 (2008) 348-351
                'alpha': 0.2,
                'beta': 0.8,
                'omega': 0.44,
                'cam': True,
                'rsf': 'Yukawa',
                'xc': 'HYB_GGA_XC_CAMY_BLYP'
            },
            'CAMY-B3LYP': {  # Seth, Ziegler JCTC 8 (2012) 901-907
                'alpha': 0.19,
                'beta': 0.46,
                'omega': 0.34,
                'cam': True,
                'rsf': 'Yukawa',
                'xc': 'HYB_GGA_XC_CAMY_B3LYP'
            },
            'LCY-BLYP': {  # Seth, Ziegler JCTC 8 (2012) 901-907
                'alpha': 0.0,
                'beta': 1.0,
                'omega': 0.75,
                'cam': False,
                'rsf': 'Yukawa',
                'xc': 'HYB_GGA_XC_LCY_BLYP'
            },
            'LCY-PBE': {  # Seth, Ziegler JCTC 8 (2012) 901-907
                'alpha': 0.0,
                'beta': 1.0,
                'omega': 0.75,
                'cam': False,
                'rsf': 'Yukawa',
                'xc': 'HYB_GGA_XC_LCY_PBE'
            }
        }
        self.omega = None
        self.cam_alpha = None
        self.cam_beta = None
        self.is_cam = False
        self.rsf = None

        def _xc(name):
            return {'name': name, 'stencil': stencil}

        if name == 'EXX':
            hybrid = 1.0
            xc = XC(XCNull())
        elif name == 'PBE0':
            hybrid = 0.25
            xc = XC(_xc('HYB_GGA_XC_PBEH'))
        elif name == 'B3LYP':
            hybrid = 0.2
            xc = XC(_xc('HYB_GGA_XC_B3LYP'))
        elif name in rsf_functionals:
            rsf_functional = rsf_functionals[name]
            self.cam_alpha = rsf_functional['alpha']
            self.cam_beta = rsf_functional['beta']
            self.omega = rsf_functional['omega']
            self.is_cam = rsf_functional['cam']
            self.rsf = rsf_functional['rsf']
            xc = XC(rsf_functional['xc'])
            hybrid = self.cam_alpha + self.cam_beta

        if isinstance(xc, (str, dict)):
            xc = XC(xc)

        self.hybrid = float(hybrid)
        self.xc = xc
        if omega is not None:
            omega = float(omega)
            if self.omega is not None and self.omega != omega:
                self.xc.kernel.set_omega(omega)
                # Needed to tune omega for RSF
            self.omega = omega
        XCFunctional.__init__(self, name, xc.type)

    def todict(self):
        return {'name': self.name,
                'hybrid': self.hybrid,
                'excitation': self.excitation,
                'excited': self.excited,
                'xc': self.xc.todict(),
                'omega': self.omega}

    def tostring(self):
        """Return string suitable to generate xc from string."""
        xc_dict = self.todict()
        for test_key in ['name', 'xc', 'kernel', 'type']:
            if test_key in xc_dict:
                del xc_dict[test_key]
        return self.name + ':' + ':'.join([(k + '=' + repr(v))
                                           for k, v in xc_dict.items()])

    def get_setup_name(self):
        return 'PBE'


class HybridXC(HybridXCBase):
    def __init__(self, name, hybrid=None, xc=None,
                 finegrid=False, unocc=False, omega=None,
                 excitation=None, excited=0, stencil=2):
        """Mix standard functionals with exact exchange.

        finegrid: boolean
            Use fine grid for energy functional evaluations ?
        unocc: boolean
            Apply vxx also to unoccupied states ?
        omega: float
            RSF mixing parameter
        excitation: string:
            Apply operator for improved virtual orbitals
            to unocc states? Possible modes:
                singlet: excitations to singlets
                triplet: excitations to triplets
                average: average between singlets and tripletts
                see f.e. https://doi.org/10.1021/acs.jctc.8b00238
        excited: number
            Band to excite from - counted from HOMO downwards

        """
        self.finegrid = finegrid
        self.unocc = unocc
        self.excitation = excitation
        self.excited = excited
        HybridXCBase.__init__(self, name, hybrid=hybrid, xc=xc, omega=omega,
                              stencil=stencil)

        # Note: self.omega may not be identical to omega!
        self.yukawa_interactions = CachedYukawaInteractions(self.omega)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dEdD_sp,
                                                addcoredensity, a)

    def initialize(self, density, hamiltonian, wfs):
        assert wfs.kd.gamma
        self.xc.initialize(density, hamiltonian, wfs)
        self.kpt_comm = wfs.kd.comm
        self.nspins = wfs.nspins
        self.setups = wfs.setups
        self.density = density
        self.kpt_u = wfs.kpt_u
        self.exx_s = np.zeros(self.nspins)
        self.ekin_s = np.zeros(self.nspins)
        self.nocc_s = np.empty(self.nspins, int)

        self.gd = density.gd
        self.redistributor = density.redistributor

        use_charge_center = hamiltonian.poisson.use_charge_center
        # XXX How do we construct a copy of the Poisson solver of the
        # Hamiltonian?  We don't know what class it is, etc., but gd
        # may differ.
        # XXX One might consider using a charged centered compensation
        # charge for the PoissonSolver in the case of EXX as standard
        self.poissonsolver = PoissonSolver(
            'fd', eps=1e-12, use_charge_center=use_charge_center)
        # self.poissonsolver = hamiltonian.poisson

        if self.finegrid:
            self.finegd = self.gd.refine()
            # XXX Taking restrictor from Hamiltonian will not work in PW mode,
            # will it?  I think this supports only real-space mode.
            # self.restrictor = hamiltonian.restrictor
            self.restrictor = Transformer(self.finegd, self.gd, 3)
            self.interpolator = Transformer(self.gd, self.finegd, 3)
        else:
            self.finegd = self.gd

        self.ghat = LFC(self.finegd,
                        [setup.ghat_l for setup in density.setups],
                        integral=np.sqrt(4 * np.pi), forces=True)
        self.poissonsolver.set_grid_descriptor(self.finegd)
        if self.rsf == 'Yukawa':
            omega2 = self.omega**2
            self.screened_poissonsolver = HelmholtzSolver(
                k2=-omega2, eps=1e-12, nn=3,
                use_charge_center=use_charge_center)
            self.screened_poissonsolver.set_grid_descriptor(self.finegd)

    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)
        # Note that the quantities passed are on the
        # density/Hamiltonian grids!
        # They may be distributed differently from own quantities.
        self.ekin = self.kpt_comm.sum_scalar(self.ekin_s.sum())
        return exc + self.kpt_comm.sum_scalar(self.exx_s.sum())

    def calculate_exx(self):
        for kpt in self.kpt_u:
            self.apply_orbital_dependent_hamiltonian(kpt, kpt.psit_nG)

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_nG,
                                            Htpsit_nG=None, dH_asp=None):
        if kpt.f_n is None:
            return

        deg = 2 // self.nspins   # Spin degeneracy
        hybrid = self.hybrid
        P_ani = kpt.P_ani
        setups = self.setups
        is_cam = self.is_cam

        vt_g = self.finegd.empty()
        if self.gd is not self.finegd:
            vt_G = self.gd.empty()
        if self.rsf == 'Yukawa':
            y_vt_g = self.finegd.empty()
            # if self.gd is not self.finegd:
            #     y_vt_G = self.gd.empty()

        nocc = int(ceil(kpt.f_n.sum())) // (3 - self.nspins)
        if self.excitation is not None:
            ex_band = nocc - self.excited - 1
            if self.excitation == 'singlet':
                ex_weight = -1
            elif self.excitation == 'triplet':
                ex_weight = +1
            else:
                ex_weight = 0

        if self.unocc or self.excitation is not None:
            nbands = len(kpt.f_n)
        else:
            nbands = nocc
        self.nocc_s[kpt.s] = nocc

        if Htpsit_nG is not None:
            kpt.vt_nG = self.gd.empty(nbands)
            kpt.vxx_ani = {}
            kpt.vxx_anii = {}
            for a, P_ni in P_ani.items():
                I = P_ni.shape[1]
                kpt.vxx_ani[a] = np.zeros((nbands, I))
                kpt.vxx_anii[a] = np.zeros((nbands, I, I))

        exx = 0.0
        ekin = 0.0

        # XXXX nbands can be different numbers on different cpus!
        # That means some will execute the loop and others not.
        # And deadlocks with augment-grids.

        # Determine pseudo-exchange
        for n1 in range(nbands):
            psit1_G = psit_nG[n1]
            f1 = kpt.f_n[n1] / deg
            for n2 in range(n1, nbands):
                psit2_G = psit_nG[n2]
                f2 = kpt.f_n[n2] / deg
                if n1 != n2 and f1 == 0 and f1 == f2:
                    continue    # Don't work on double unocc. bands
                # Double count factor:
                dc = (1 + (n1 != n2)) * deg
                nt_G, rhot_g = self.calculate_pair_density(n1, n2, psit_nG,
                                                           P_ani)
                vt_g[:] = 0.0
                # XXXXX This will go wrong because we are solving the
                # Poisson equation on the distribution of gd, not finegd
                # Or maybe it's fixed now

                self.poissonsolver.solve(vt_g, -rhot_g,
                                         charge=-float(n1 == n2),
                                         zero_initial_phi=True)
                vt_g *= hybrid
                if self.rsf == 'Yukawa':
                    y_vt_g[:] = 0.0
                    self.screened_poissonsolver.solve(
                        y_vt_g, -rhot_g, charge=-float(n1 == n2),
                        zero_initial_phi=True)
                    if is_cam:  # Cam like correction
                        y_vt_g *= self.cam_beta
                    else:
                        y_vt_g *= hybrid
                    vt_g -= y_vt_g
                if self.gd is self.finegd:
                    vt_G = vt_g
                else:
                    self.restrictor.apply(vt_g, vt_G)

                # Integrate the potential on fine and coarse grids
                int_fine = self.finegd.integrate(vt_g * rhot_g)
                int_coarse = self.gd.integrate(vt_G * nt_G)
                if self.gd.comm.rank == 0:  # only add to energy on master CPU
                    exx += 0.5 * dc * f1 * f2 * int_fine
                    ekin -= dc * f1 * f2 * int_coarse
                if Htpsit_nG is not None:
                    Htpsit_nG[n1] += f2 * vt_G * psit2_G
                    if n1 == n2:
                        kpt.vt_nG[n1] = f1 * vt_G
                        if self.excitation is not None and n1 == ex_band:
                            Htpsit_nG[nocc:] += f1 * vt_G * psit_nG[nocc:]
                    else:
                        if self.excitation is None or n1 != ex_band \
                                or n2 < nocc:
                            Htpsit_nG[n2] += f1 * vt_G * psit1_G
                        else:
                            Htpsit_nG[n2] += f1 * ex_weight * vt_G * psit1_G

                    # Update the vxx_uni and vxx_unii vectors of the nuclei,
                    # used to determine the atomic hamiltonian, and the
                    # residuals
                    v_aL = self.ghat.dict()
                    self.ghat.integrate(vt_g, v_aL)
                    for a, v_L in v_aL.items():
                        v_ii = unpack_hermitian(
                            np.dot(setups[a].Delta_pL, v_L))
                        v_ni = kpt.vxx_ani[a]
                        v_nii = kpt.vxx_anii[a]
                        P_ni = P_ani[a]
                        v_ni[n1] += f2 * np.dot(v_ii, P_ni[n2])
                        if n1 != n2:
                            if self.excitation is None or n1 != ex_band or \
                                    n2 < nocc:
                                v_ni[n2] += f1 * np.dot(v_ii, P_ni[n1])
                            else:
                                v_ni[n2] += f1 * ex_weight * \
                                    np.dot(v_ii, P_ni[n1])
                        else:
                            # XXX Check this:
                            v_nii[n1] = f1 * v_ii
                            if self.excitation is not None and n1 == ex_band:
                                for nuoc in range(nocc, nbands):
                                    v_ni[nuoc] += f1 * \
                                        np.dot(v_ii, P_ni[nuoc])

        def calculate_vv(ni, D_ii, M_pp, weight, addme=False):
            """Calculate the local corrections depending on Mpp."""
            dexx = 0
            dekin = 0
            if not addme:
                addsign = -2.0
            else:
                addsign = 2.0
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += M_pp[p13, p24] * D_ii[i3, i4]
                    p12 = packed_index(i1, i2, ni)
                    if Htpsit_nG is not None:
                        dH_p[p12] += addsign * weight / \
                            deg * A / ((i1 != i2) + 1)
                    dekin += 2 * weight / deg * D_ii[i1, i2] * A
                    dexx -= weight / deg * D_ii[i1, i2] * A
            return (dexx, dekin)

        # Apply the atomic corrections to the energy and the Hamiltonian
        # matrix
        for a, P_ni in P_ani.items():
            setup = setups[a]

            if Htpsit_nG is not None:
                # Add non-trivial corrections the Hamiltonian matrix
                h_nn = symmetrize(np.inner(P_ni[:nbands],
                                           kpt.vxx_ani[a][:nbands]))
                ekin -= np.dot(kpt.f_n[:nbands], h_nn.diagonal())

                dH_p = dH_asp[a][kpt.s]

            # Get atomic density and Hamiltonian matrices
            D_p = self.density.D_asp[a][kpt.s]
            D_ii = unpack_density(D_p)
            ni = len(D_ii)

            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            (dexx, dekin) = calculate_vv(ni, D_ii, setup.M_pp, hybrid)
            ekin += dekin
            exx += dexx

            if self.rsf is not None:
                Mg_pp = self.yukawa_interactions.get_Mg_pp(setup)
                if is_cam:
                    (dexx, dekin) = calculate_vv(
                        ni, D_ii, Mg_pp, self.cam_beta, addme=True)
                else:
                    (dexx, dekin) = calculate_vv(
                        ni, D_ii, Mg_pp, hybrid, addme=True)
                ekin -= dekin
                exx -= dexx
            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            if setup.X_p is not None:
                exx -= hybrid * np.dot(D_p, setup.X_p)
                if Htpsit_nG is not None:
                    dH_p -= hybrid * setup.X_p
                    ekin += hybrid * np.dot(D_p, setup.X_p)

                if self.rsf == 'Yukawa' and setup.X_pg is not None:
                    if is_cam:
                        thybrid = self.cam_beta  # 0th order
                    else:
                        thybrid = hybrid
                    exx += thybrid * np.dot(D_p, setup.X_pg)
                    if Htpsit_nG is not None:
                        dH_p += thybrid * setup.X_pg
                        ekin -= thybrid * np.dot(D_p, setup.X_pg)
                elif self.rsf == 'Yukawa' and setup.X_pg is None:
                    thybrid = exp(-3.62e-2 * self.omega)  # educated guess
                    if is_cam:
                        thybrid *= self.cam_beta
                    else:
                        thybrid *= hybrid
                    exx += thybrid * np.dot(D_p, setup.X_p)
                    if Htpsit_nG is not None:
                        dH_p += thybrid * setup.X_p
                        ekin -= thybrid * np.dot(D_p, setup.X_p)
                # Add core-core exchange energy
                if kpt.s == 0:
                    if self.rsf is None or is_cam:
                        if is_cam:
                            exx += self.cam_alpha * setup.ExxC
                        else:
                            exx += hybrid * setup.ExxC

        self.exx_s[kpt.s] = self.gd.comm.sum_scalar(exx)
        self.ekin_s[kpt.s] = self.gd.comm.sum_scalar(ekin)

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        if not hasattr(kpt, 'vxx_ani'):
            return

        # if self.gd.comm.rank > 0:
        #    H_nn[:] = 0.0

        nocc = self.nocc_s[kpt.s]
        nbands = len(kpt.vt_nG)
        for a, P_ni in kpt.P_ani.items():
            H_nn[:nbands, :nbands] += symmetrize(np.inner(P_ni[:nbands],
                                                          kpt.vxx_ani[a]))
        # self.gd.comm.sum(H_nn)

        if not self.unocc or self.excitation is not None:
            H_nn[:nocc, nocc:] = 0.0
            H_nn[nocc:, :nocc] = 0.0

    def calculate_pair_density(self, n1, n2, psit_nG, P_ani):
        Q_aL = {}
        for a, P_ni in P_ani.items():
            P1_i = P_ni[n1]
            P2_i = P_ni[n2]
            D_ii = np.outer(P1_i, P2_i.conj()).real
            D_p = pack_density(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        nt_G = psit_nG[n1] * psit_nG[n2]

        if self.finegd is self.gd:
            nt_g = nt_G
        else:
            nt_g = self.finegd.empty()
            self.interpolator.apply(nt_G, nt_g)

        rhot_g = nt_g.copy()
        self.ghat.add(rhot_g, Q_aL)

        return nt_G, rhot_g

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        if kpt.f_n is None:
            return

        if self.unocc or self.excitation is not None:
            nocc = len(kpt.vt_nG)
        else:
            nocc = self.nocc_s[kpt.s]

        if calculate_change:
            for x, n in enumerate(n_x):
                if n < nocc:
                    Htpsit_xG[x] += kpt.vt_nG[n] * psit_xG[x]
                    for a, P_xi in P_axi.items():
                        c_axi[a][x] += np.dot(kpt.vxx_anii[a][n], P_xi[x])
        else:
            for a, c_xi in c_axi.items():
                c_xi[:nocc] += kpt.vxx_ani[a][:nocc]

    def rotate(self, kpt, U_nn):
        if kpt.f_n is None:
            return

        U_nn = U_nn.T.copy()
        nocc = self.nocc_s[kpt.s]
        if len(kpt.vt_nG) == nocc:
            U_nn = U_nn[:nocc, :nocc]
        # gemm(1.0, kpt.vt_nG.copy(), U_nn, 0.0, kpt.vt_nG)
        n, G1, G2, G3 = kpt.vt_nG.shape
        vt_nG = kpt.vt_nG.reshape((n, G1 * G2 * G3))
        mmm(1.0, U_nn, 'N', vt_nG.copy(), 'N', 0.0, vt_nG)
        for v_ni in kpt.vxx_ani.values():
            # gemm(1.0, v_ni.copy(), U_nn, 0.0, v_ni)
            mmm(1.0, U_nn, 'N', v_ni.copy(), 'N', 0.0, v_ni)
        for v_nii in kpt.vxx_anii.values():
            # gemm(1.0, v_nii.copy(), U_nn, 0.0, v_nii)
            n, i, i = v_nii.shape
            v_nii = v_nii.reshape((n, i**2))
            mmm(1.0, U_nn, 'N', v_nii.copy(), 'N', 0.0, v_nii)


def atomic_exact_exchange(atom, type='all'):
    """Returns the exact exchange energy of the atom defined by the
       instantiated AllElectron object 'atom'
    """
    G_LLL = gaunt(lmax=max(atom.l_j))  # Make gaunt coeff. list
    Nj = len(atom.n_j)  # The total number of orbitals

    # determine relevant states for chosen type of exchange contribution
    if type == 'all':
        nstates = mstates = range(Nj)
    else:
        Njcore = core_states(atom.symbol)  # The number of core orbitals
        if type == 'val-val':
            nstates = mstates = range(Njcore, Nj)
        elif type == 'core-core':
            nstates = mstates = range(Njcore)
        elif type == 'val-core':
            nstates = range(Njcore, Nj)
            mstates = range(Njcore)
        else:
            raise RuntimeError('Unknown type of exchange: ', type)

    # Arrays for storing the potential (times radius)
    vr = np.zeros(atom.N)
    vrl = np.zeros(atom.N)

    # do actual calculation of exchange contribution
    Exx = 0.0
    for j1 in nstates:
        # angular momentum of first state
        l1 = atom.l_j[j1]

        for j2 in mstates:
            # angular momentum of second state
            l2 = atom.l_j[j2]

            # joint occupation number
            f12 = 0.5 * (atom.f_j[j1] / (2. * l1 + 1) *
                         atom.f_j[j2] / (2. * l2 + 1))

            # electron density times radius times length element
            nrdr = atom.u_j[j1] * atom.u_j[j2] * atom.dr
            nrdr[1:] /= atom.r[1:]

            # potential times radius
            vr[:] = 0.0

            # L summation
            for l in range(l1 + l2 + 1):
                # get potential for current l-value
                hartree(l, nrdr, atom.r, vrl)

                # take all m1 m2 and m values of Gaunt matrix of the form
                # G(L1,L2,L) where L = {l,m}
                G2 = G_LLL[l1**2:(l1 + 1)**2,
                           l2**2:(l2 + 1)**2,
                           l**2:(l + 1)**2]**2

                # add to total potential
                vr += vrl * np.sum(G2)

            # add to total exchange the contribution from current two states
            Exx += -.5 * f12 * np.dot(vr, nrdr)

    # double energy if mixed contribution
    if type == 'val-core':
        Exx *= 2.

    # return exchange energy
    return Exx


def constructX(gen, gamma=0):
    """Construct the X_p^a matrix for the given atom.

    The X_p^a matrix describes the valence-core interactions of the
    partial waves.
    """
    # initialize attributes
    uv_j = gen.vu_j    # soft valence states * r:
    lv_j = gen.vl_j    # their repective l quantum numbers
    Nvi = 0
    for l in lv_j:
        Nvi += 2 * l + 1   # total number of valence states (including m)

    # number of core and valence orbitals (j only, i.e. not m-number)
    Njcore = gen.njcore
    Njval = len(lv_j)

    # core states * r:
    uc_j = gen.u_j[:Njcore]
    r, dr, N = gen.r, gen.dr, gen.N
    r2 = r**2

    # potential times radius
    vr = np.zeros(N)

    # initialize X_ii matrix
    X_ii = np.zeros((Nvi, Nvi))

    # make gaunt coeff. list
    lmax = max(gen.l_j[:Njcore] + gen.vl_j)
    G_LLL = gaunt(lmax=lmax)

    # sum over core states
    for jc in range(Njcore):
        lc = gen.l_j[jc]

        # sum over first valence state index
        i1 = 0
        for jv1 in range(Njval):
            lv1 = lv_j[jv1]

            # electron density 1 times radius times length element
            n1c = uv_j[jv1] * uc_j[jc] * dr
            n1c[1:] /= r[1:]

            # sum over second valence state index
            i2 = 0
            for jv2 in range(Njval):
                lv2 = lv_j[jv2]

                # electron density 2
                n2c = uv_j[jv2] * uc_j[jc]
                n2c[1:] /= r2[1:]

                # sum expansion in angular momenta
                for l in range(min(lv1, lv2) + lc + 1):
                    # Int density * potential * r^2 * dr:
                    if gamma == 0:
                        vr = gen.rgd.poisson(n2c, l)
                    else:
                        vr = gen.rgd.yukawa(n2c, l, gamma)
                    nv = np.dot(n1c, vr)

                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2 * lc + 1):
                        for m in range(2 * l + 1):
                            G1c = G_LLL[lv1**2:(lv1 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            G2c = G_LLL[lv2**2:(lv2 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            A_mm += nv * np.outer(G1c, G2c)
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack_hermitian(X_ii)
    return X_p


def H_coulomb_val_core(paw, u=0):
    """Short description here.

    ::

                     core   *    *
             //       --   i(r) k(r') k(r) j (r')
       H   = || drdr' >   ----------------------
        ij   //       --          |r - r'|
                      k
    """
    H_nn = np.zeros((paw.wfs.bd.nbands, paw.wfs.bd.nbands),
                    dtype=paw.wfs.dtype)
    for a, P_ni in paw.wfs.kpt_u[u].P_ani.items():
        X_ii = unpack_hermitian(paw.wfs.setups[a].X_p)
        H_nn += np.dot(P_ni.conj(), np.dot(X_ii, P_ni.T))
    paw.wfs.gd.comm.sum(H_nn)
    from ase.units import Hartree
    return H_nn * Hartree
