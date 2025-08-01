from math import pi

import numpy as np

from my_gpaw25.xc.lda import calculate_paw_correction
from my_gpaw25.utilities.blas import axpy
from my_gpaw25.fd_operators import Gradient
from my_gpaw25.sphere.lebedev import Y_nL, weight_n
from my_gpaw25.xc.pawcorrection import rnablaY_nLv
from my_gpaw25.xc.functional import XCFunctional


class GGARadialExpansion:
    def __init__(self, rcalc, *args):
        self.rcalc = rcalc
        self.args = args

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg

        dndr_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg in zip(n_sLg, dndr_sLg):
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                rgd.derivative(n_g, dndr_g)

        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))

        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            e_g, dedn_sg, b_vsg, dedsigma_xg = \
                self.rcalc(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n,
                           *self.args)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            dedsigma_xg *= rgd.dr_g
            B_vsg = dedsigma_xg[::2] * b_vsg
            if nspins == 2:
                B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
            B_vsq = np.dot(B_vsg, n_qg.T)
            dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
            E += w * rgd.integrate(e_g)

        return E, dEdD_sqL


# First part of gga_calculate_radial - initializes some quantities.
def radial_gga_vars(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv):
    nspins = len(n_sLg)

    n_sg = np.dot(Y_L, n_sLg)

    a_sg = np.dot(Y_L, dndr_sLg)
    b_vsg = np.dot(rnablaY_Lv.T, n_sLg)

    sigma_xg = rgd.empty(2 * nspins - 1)
    sigma_xg[::2] = (b_vsg ** 2).sum(0)
    if nspins == 2:
        sigma_xg[1] = (b_vsg[:, 0] * b_vsg[:, 1]).sum(0)
    sigma_xg[:, 1:] /= rgd.r_g[1:] ** 2
    sigma_xg[:, 0] = sigma_xg[:, 1]
    sigma_xg[::2] += a_sg ** 2
    if nspins == 2:
        sigma_xg[1] += a_sg[0] * a_sg[1]

    e_g = rgd.empty()
    dedn_sg = rgd.zeros(nspins)
    dedsigma_xg = rgd.zeros(2 * nspins - 1)
    return e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg


def add_radial_gradient_correction(rgd, sigma_xg, dedsigma_xg, a_sg):
    nspins = len(a_sg)
    vv_sg = sigma_xg[:nspins]  # reuse array
    for s in range(nspins):
        rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[2 * s] * a_sg[s],
                        vv_sg[s])
    if nspins == 2:
        v_g = sigma_xg[2]
        rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[1], v_g)
        vv_sg[0] -= v_g
        rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[0], v_g)
        vv_sg[1] -= v_g

    vv_sg[:, 1:] /= rgd.dv_g[1:]
    vv_sg[:, 0] = vv_sg[:, 1]
    return vv_sg


class GGARadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg,
         b_vsg) = radial_gga_vars(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv)
        self.kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        vv_sg = add_radial_gradient_correction(rgd, sigma_xg,
                                               dedsigma_xg, a_sg)
        return e_g, dedn_sg + vv_sg, b_vsg, dedsigma_xg


def calculate_sigma(gd, grad_v, n_sg):
    r"""Calculate sigma(r) and grad n(r).
                  _     __   _  2     __    _
    Returns sigma(r) = |\/ n(r)|  and \/ n (r).

    With multiple spins, sigma has the three elements

                   _     __     _  2
            sigma (r) = |\/ n  (r)|  ,
                 0           up

                   _     __     _      __     _
            sigma (r) =  \/ n  (r)  .  \/ n  (r) ,
                 1           up            dn

                   _     __     _  2
            sigma (r) = |\/ n  (r)|  .
                 2           dn
    """
    nspins = len(n_sg)
    gradn_svg = gd.empty((nspins, 3))
    sigma_xg = gd.zeros(nspins * 2 - 1)
    for v in range(3):
        for s in range(nspins):
            grad_v[v](n_sg[s], gradn_svg[s, v])
            axpy(1.0, gradn_svg[s, v]**2, sigma_xg[2 * s])
        if nspins == 2:
            axpy(1.0, gradn_svg[0, v] * gradn_svg[1, v], sigma_xg[1])
    return sigma_xg, gradn_svg


def add_gradient_correction(grad_v, gradn_svg, sigma_xg, dedsigma_xg, v_sg):
    r"""Add gradient correction to potential.

    ::

                      __   /    de(r)    __      \
        v  (r) += -2  \/ . |  ---------  \/ n(r) |
         xc                \  dsigma(r)          /

    Adds arbitrary data to sigma_xg.  Be sure to pass a copy if you need
    sigma_xg after calling this function.
    """
    nspins = len(v_sg)
    # vv_g is a calculation buffer.  Its contents will be corrupted.
    vv_g = sigma_xg[0]
    for v in range(3):
        for s in range(nspins):
            grad_v[v](dedsigma_xg[2 * s] * gradn_svg[s, v], vv_g)
            vv_g *= 2.0
            v_sg[s] -= vv_g
            # axpy(-2.0, vv_g, v_sg[s])
            if nspins == 2:
                grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                v_sg[1 - s] -= vv_g
                # axpy(-1.0, vv_g, v_sg[1 - s])
                # TODO: can the number of gradient evaluations be reduced?


def gga_vars(gd, grad_v, n_sg):
    nspins = len(n_sg)
    sigma_xg, gradn_svg = calculate_sigma(gd, grad_v, n_sg)
    dedsigma_xg = gd.empty(nspins * 2 - 1)
    return sigma_xg, dedsigma_xg, gradn_svg


def get_gradient_ops(gd, nn, xp):
    return [Gradient(gd, v, n=nn, xp=xp).apply for v in range(3)]


class GGA(XCFunctional):
    def __init__(self, kernel, stencil=2, xp=np):
        XCFunctional.__init__(self, kernel.name, kernel.type)
        self.kernel = kernel
        self.stencil_range = stencil
        self.xp = xp

    def set_grid_descriptor(self, gd):
        XCFunctional.set_grid_descriptor(self, gd)
        self.grad_v = get_gradient_ops(gd, self.stencil_range, self.xp)

    def todict(self):
        d = super().todict()
        d['stencil'] = self.stencil_range
        return d

    def get_description(self):
        return ('{} with {} nearest neighbor stencil'
                .format(self.name, self.stencil_range))

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg,
                                dedsigma_xg, v_sg)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        rcalc = GGARadialCalculator(self.kernel)
        expansion = GGARadialExpansion(rcalc)
        return calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        addcoredensity, a)

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        sigma_xg, gradn_svg = calculate_sigma(self.gd, self.grad_v, n_sg)
        nspins = len(n_sg)
        dedsigma_xg = self.gd.empty(nspins * 2 - 1)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)

        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                               gradn_svg[0, v2],
                                               dedsigma_xg[0]) * 2
                if nspins == 2:
                    stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[1]) * 2
                    stress_vv[v1, v2] -= integrate(gradn_svg[1, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[2]) * 2
        if not skip_sum:
            self.gd.comm.sum(stress_vv)
        return stress_vv

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)
        if e_g is None:
            e_g = rgd.empty()

        rcalc = GGARadialCalculator(self.kernel)

        e_g[:], dedn_sg = rcalc(rgd, n_sg[:, np.newaxis],
                                [1.0],
                                dndr_sg[:, np.newaxis],
                                np.zeros((1, 3)), n=None)[:2]
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)


class PurePythonGGAKernel:
    def __init__(self, name):
        self.type = 'GGA'
        self.name, self.kappa, self.mu, self.beta = pbe_constants(name)

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg, dedsigma_xg,
                  tau_sg=None, dedtau_sg=None):
        e_g[:] = 0.
        dedsigma_xg[:] = 0.

        # spin-paired:
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            # exchange
            res = gga_x(self.name, 0, n, sigma_xg[0], self.kappa, self.mu)
            ex, rs, dexdrs, dexda2 = res
            # correlation
            res = gga_c(self.name, 0, n, sigma_xg[0], 0, self.beta)
            ec, rs_, decdrs, decda2, decdzeta = res

            e_g[:] += n * (ex + ec)
            dedn_sg[:] += ex + ec - rs * (dexdrs + decdrs) / 3.
            dedsigma_xg[:] += n * (dexda2 + decda2)

        # spin-polarized:
        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40

            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40

            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            # exchange
            exa, rsa, dexadrs, dexada2 = gga_x(
                self.name, 1, na, 4.0 * sigma_xg[0], self.kappa, self.mu)
            exb, rsb, dexbdrs, dexbda2 = gga_x(
                self.name, 1, nb, 4.0 * sigma_xg[2], self.kappa, self.mu)
            a2 = sigma_xg[0] + 2.0 * sigma_xg[1] + sigma_xg[2]
            # correlation
            ec, rs, decdrs, decda2, decdzeta = gga_c(
                self.name, 1, n, a2, zeta, self.beta)

            e_g[:] += 0.5 * (na * exa + nb * exb) + n * ec
            dedn_sg[0][:] += (exa + ec - (rsa * dexadrs + rs * decdrs) / 3.0
                              - (zeta - 1.0) * decdzeta)
            dedn_sg[1][:] += (exb + ec - (rsb * dexbdrs + rs * decdrs) / 3.0
                              - (zeta + 1.0) * decdzeta)
            dedsigma_xg[0][:] += 2.0 * na * dexada2 + n * decda2
            dedsigma_xg[1][:] += 2.0 * n * decda2
            dedsigma_xg[2][:] += 2.0 * nb * dexbda2 + n * decda2


def pbe_constants(name):
    if name == 'pyPBE':
        name = 'PBE'
        kappa = 0.804
        mu = 0.2195149727645171
        beta = 0.06672455060314922
    elif name in ['pyPBEsol', 'pyzvPBEsol']:
        name = name[2:]
        kappa = 0.804
        mu = 10. / 81.
        beta = 0.046
    elif name == 'pyRPBE':
        name = 'RPBE'
        kappa = 0.804
        mu = 0.2195149727645171
        beta = 0.06672455060314922
    else:
        raise NotImplementedError(name)

    return name, kappa, mu, beta


# a2 = |grad n|^2
def gga_x(name, spin, n, a2, kappa, mu, dedmu_g=None):
    # if dedmu is given, calculate also d(e_x)/d(mu)
    assert spin in [0, 1]

    C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA = gga_constants()
    rs = (C0I / n)**(1 / 3.)

    # lda part
    ex = C1 / rs
    dexdrs = -ex / rs

    # gga part
    c = (C2 * rs / n)**2.
    s2 = a2 * c

    if name in ['PBE', 'PBEsol', 'zvPBEsol', 'QNA']:
        x = 1.0 + mu * s2 / kappa
        Fx = 1.0 + kappa - kappa / x
        dFxds2 = mu / (x**2.)
    elif name == 'RPBE':
        arg = np.maximum(-mu * s2 / kappa, -5.e2)
        x = np.exp(arg)
        Fx = 1.0 + kappa * (1.0 - x)
        dFxds2 = mu * x
    else:
        raise NotImplementedError

    ds2drs = 8.0 * c * a2 / rs
    dexdrs = dexdrs * Fx + ex * dFxds2 * ds2drs
    dexda2 = ex * dFxds2 * c
    if dedmu_g is not None:
        dedmu_g[:] = ex * s2 / (1 + mu * s2 / kappa)**2
    ex *= Fx

    return ex, rs, dexdrs, dexda2


def gga_c(name, spin, n, a2, zeta, BETA, decdbeta_g=None):
    # If decdbeta is specified, calculate also d(e_c)/d(beta)
    assert spin in [0, 1]
    from my_gpaw25.xc.lda import G

    C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA = gga_constants()
    rs = (C0I / n)**(1 / 3.)

    if name == 'zvPBEsol':
        zv_a = 1.8
        zv_o = 9. / 2.
        zv_x = 1. / 6.

    # lda part
    ec, decdrs_0 = G(rs**0.5, 0.031091, 0.21370, 7.5957,
                     3.5876, 1.6382, 0.49294)

    if spin == 0:
        decdrs = decdrs_0
        decdzeta = 0.  # dummy
    else:
        e1, decdrs_1 = G(rs**0.5, 0.015545, 0.20548, 14.1189,
                         6.1977, 3.3662, 0.62517)
        alpha, dalphadrs = G(rs**0.5, 0.016887, 0.11125, 10.357,
                             3.6231, 0.88026, 0.49671)
        alpha *= -1.
        dalphadrs *= -1.
        zp = 1.0 + zeta
        zm = 1.0 - zeta
        xp = zp**(1 / 3.)
        xm = zm**(1 / 3.)
        f = CC1 * (zp * xp + zm * xm - 2.0)
        f1 = CC2 * (xp - xm)
        zeta3 = zeta * zeta * zeta
        zeta4 = zeta * zeta * zeta * zeta
        x = 1.0 - zeta4
        decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                  decdrs_1 * f * zeta4 +
                  dalphadrs * f * x * IF2)
        decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                    f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
        ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4

    # gga part
    n2 = n * n
    if spin == 1:
        phi = 0.5 * (xp * xp + xm * xm)
        phi2 = phi * phi
        phi3 = phi * phi2
        t2 = C3 * a2 * rs / (n2 * phi2)
        y = -ec / (GAMMA * phi3)
        if name == 'zvPBEsol':
            u3 = t2**(3. / 2.) * phi3 * (rs / 3.)**(-3. * zv_x)
            zvarg = -zv_a * u3 * abs(zeta)**zv_o
            zvf = np.exp(zvarg)
    else:
        t2 = C3 * a2 * rs / n2
        y = -ec / GAMMA

    x = np.exp(y)

    A = np.zeros_like(x)
    indices = np.nonzero(y)
    if isinstance(BETA, np.ndarray):
        A[indices] = (BETA[indices] / (GAMMA * (x[indices] - 1.0)))
    else:
        A[indices] = (BETA / (GAMMA * (x[indices] - 1.0)))

    At2 = A * t2
    nom = 1.0 + At2
    denom = nom + At2 * At2
    X = 1.0 + BETA * t2 * nom / (denom * GAMMA)
    H = GAMMA * np.log(X)
    tmp = (GAMMA * BETA / (denom * (BETA * t2 * nom + GAMMA * denom)))
    tmp2 = A * A * x / BETA
    dAdrs = tmp2 * decdrs
    if spin == 1:
        H *= phi3
        tmp *= phi3
        dAdrs /= phi3
        if name == 'zvPBEsol':
            H_ = H.copy()
            H *= zvf
    dHdt2 = (1.0 + 2.0 * At2) * tmp
    dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp
    decdrs += dHdt2 * 7.0 * t2 / rs + dHdA * dAdrs
    decda2 = dHdt2 * C3 * rs / n2
    if spin == 1:
        dphidzeta = np.zeros_like(x)
        ind1 = np.nonzero(xp)
        ind2 = np.nonzero(xm)
        dphidzeta[ind1] += 1.0 / (3.0 * xp[ind1])
        dphidzeta[ind2] -= 1.0 / (3.0 * xm[ind2])
        dAdzeta = tmp2 * (decdzeta - 3.0 * ec * dphidzeta / phi) / phi3
        decdzeta += ((3.0 * H / phi - dHdt2 * 2.0 * t2 / phi) * dphidzeta
                     + dHdA * dAdzeta)
        decda2 /= phi2
        if name == 'zvPBEsol':
            u3_ = (t2**(3. / 2.) *
                   phi3 * rs**(-3. * zv_x - 1.) / 3.**(-3. * zv_x))
            zvarg_ = -zv_a * u3_ * abs(zeta)**zv_o
            dzvfdrs = -3. * zv_x * zvf * zvarg_
            decdrs *= zvf
            decdrs += H_ * dzvfdrs

            dt2da2 = C3 * rs / n2
            assert np.shape(dt2da2) == np.shape(t2)
            dzvfda2 = dt2da2 * zvf * zvarg * 3. / (2. * t2)
            decda2 *= zvf
            decda2 += H_ * dzvfda2

            dadz = zv_o * abs(zeta)**(zv_o - 1.)
            dbdphi = 3. * u3 / phi
            dPdzeta = u3 * dadz + abs(zeta)**(zv_o) * dbdphi * dphidzeta
            dzvfdzeta = -zv_a * zvf * dPdzeta
            decdzeta *= zvf
            decdzeta += H_ * dzvfdzeta
    ec += H

    if decdbeta_g is not None:
        if spin == 0:
            phi3 = 1.0
        Y = GAMMA * phi3
        decdbeta_g[:] = Y / X * t2 / GAMMA
        decdbeta_g *= ((1 + 2 * At2) / (1 + At2 + At2**2) -
                       (1 + At2) * (At2 + 2 * At2**2) / (1 + At2 + At2**2)**2)

    return ec, rs, decdrs, decda2, decdzeta


def gga_constants():
    from my_gpaw25.xc.lda import lda_constants
    C0I, C1, CC1, CC2, IF2 = lda_constants()
    C2 = 0.26053088059892404
    C3 = 0.10231023756535741
    GAMMA = 0.0310906908697

    return C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA
