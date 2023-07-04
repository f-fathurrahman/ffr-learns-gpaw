import numpy as np

from my_gpaw.pw.lfc import ft
from my_gpaw.gaunt import gaunt
from my_gpaw.spherical_harmonics import Y
from types import SimpleNamespace


class Setuplet:
    def __init__(self, *, phit_jg, phi_jg, rgd, l_j, rcut_j):
        self.rgd = rgd
        self.data = SimpleNamespace(phit_jg=phit_jg, phi_jg=phi_jg)
        self.l_j = l_j
        self.ni = np.sum([2 * l + 1 for l in l_j])
        self.rcut_j = rcut_j


def two_phi_planewave_integrals(qG_Gv, *, pawdata):
    rgd = pawdata.rgd
    l_j = pawdata.l_j
    phi_jg = pawdata.data.phi_jg
    phit_jg = pawdata.data.phit_jg
    ni = pawdata.ni
    
    gcut2 = rgd.ceil(2 * max(pawdata.rcut_j))
    
    # Initialize
    npw = qG_Gv.shape[0]
    phi_Gii = np.zeros((npw, ni, ni), dtype=complex)

    G_LLL = gaunt(max(l_j))
    k_G = np.sum(qG_Gv**2, axis=1)**0.5

    i1_start = 0

    for j1, l1 in enumerate(l_j):
        i2_start = 0
        for j2, l2 in enumerate(l_j):
            # Calculate the radial part of the product density
            rhot_g = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                spline = rgd.spline(rhot_g[:gcut2], l=l, points=2**10)
                splineG = ft(spline, N=2**12)
                f_G = splineG.map(k_G) * (-1j)**l

                for m1 in range(2 * l1 + 1):
                    i1 = i1_start + m1
                    for m2 in range(2 * l2 + 1):
                        i2 = i2_start + m2
                        G_m = G_LLL[l1**2 + m1, l2**2 + m2, l**2:(l + 1)**2]
                        for m, G in enumerate(G_m):
                            # If Gaunt coefficient is zero, no need to add
                            if G == 0:
                                continue
                            x_G = Y(l**2 + m, *qG_Gv.T) * f_G
                            phi_Gii[:, i1, i2] += G * x_G

            i2_start += 2 * l2 + 1
        i1_start += 2 * l1 + 1
    return phi_Gii.reshape(npw, ni * ni)


class PWPAWCorrectionData:
    def __init__(self, Q_aGii, qpd, pawdatasets, pos_av):
        # Sometimes we loop over these in ways that are very dangerous.
        # It must be list, not dictionary.
        assert isinstance(Q_aGii, list)
        assert len(Q_aGii) == len(pos_av) == len(pawdatasets)

        self.Q_aGii = Q_aGii

        self.qpd = qpd
        self.pawdatasets = pawdatasets
        self.pos_av = pos_av

    def _new(self, Q_aGii):
        return PWPAWCorrectionData(Q_aGii, qpd=self.qpd,
                                   pawdatasets=self.pawdatasets,
                                   pos_av=self.pos_av)

    def remap(self, M_vv, G_Gv, sym, sign):
        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = self._get_x_G(G_Gv, M_vv, self.pos_av[a])
            U_ii = self.pawdatasets[a].R_sii[sym]

            Q_Gii = np.einsum('ij,kjl,ml->kim',
                              U_ii,
                              Q_Gii * x_G[:, None, None],
                              U_ii,
                              optimize='optimal')
            if sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        return self._new(Q_aGii)

    def _get_x_G(self, G_Gv, M_vv, pos_v):
        # This doesn't really belong here.  Or does it?  Maybe this formula
        # is only used with PAW corrections.
        return np.exp(1j * (G_Gv @ (pos_v - M_vv @ pos_v)))

    def remap_by_symop(self, symop, G_Gv, M_vv):
        return self.remap(M_vv, G_Gv, symop.symno, symop.sign)

    def multiply(self, P_ani, band):
        assert isinstance(P_ani, list)
        assert len(P_ani) == len(self.Q_aGii)

        C1_aGi = [Qa_Gii @ P1_ni[band].conj()
                  for Qa_Gii, P1_ni in zip(self.Q_aGii, P_ani)]
        return C1_aGi

    def reduce_ecut(self, G2G):
        # XXX actually we should return this with another PW descriptor.
        return self._new([Q_Gii.take(G2G, axis=0) for Q_Gii in self.Q_aGii])

    def almost_equal(self, otherpawcorr, G_G):
        for a, Q_Gii in enumerate(otherpawcorr.Q_aGii):
            e = abs(self.Q_aGii[a] - Q_Gii[G_G]).max()
            if e > 1e-12:
                return False
        return True


def get_pair_density_paw_corrections(pawdatasets, qpd, spos_ac):
    """Calculate and bundle paw corrections to the pair densities as a
    PWPAWCorrectionData object."""
    qG_Gv = qpd.get_reciprocal_vectors(add_q=True)
    pos_av = spos_ac @ qpd.gd.cell_cv

    # Collect integrals for all species:
    Q_xGii = {}
    for atom_index, pawdata in enumerate(pawdatasets):
        Q_Gii = two_phi_planewave_integrals(qG_Gv, pawdata=pawdata)
        ni = pawdata.ni
        Q_xGii[atom_index] = Q_Gii.reshape(-1, ni, ni)

    Q_aGii = []
    for atom_index, pawdata in enumerate(pawdatasets):
        Q_Gii = Q_xGii[atom_index]
        x_G = np.exp(-1j * (qG_Gv @ pos_av[atom_index]))
        Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)

    return PWPAWCorrectionData(Q_aGii, qpd=qpd,
                               pawdatasets=pawdatasets,
                               pos_av=pos_av)
