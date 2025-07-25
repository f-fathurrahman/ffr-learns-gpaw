import numpy as np
from scipy.special import erf

from my_gpaw25.atom.atompaw import AtomPAW
from my_gpaw25.atom.radialgd import EquidistantRadialGridDescriptor
from my_gpaw25.basis_data import Basis, BasisFunction
from my_gpaw25.setup import BaseSetup, LocalCorrectionVar
from my_gpaw25.spline import Spline
from my_gpaw25.utilities import divrl, hartree as hartree_solve


null_spline = Spline.from_data(0, 1.0, [0., 0., 0.])


# XXX Not used at the moment; see comment below about rgd splines.
def projectors_to_splines(rgd, l_j, pt_jg, filter=None):
    # This function exists because both HGH and SG15 needs to do
    # exactly the same thing.
    #
    # XXX equal-range projectors still required for some reason
    maxlen = max([len(pt_g) for pt_g in pt_jg])
    pt_j = []
    for l, pt1_g in zip(l_j, pt_jg):
        pt2_g = np.zeros(maxlen)
        pt2_g[:len(pt1_g)] = pt1_g
        if filter is not None:
            filter(rgd, rgd.r_g[maxlen], pt2_g, l=l)
        pt2_g = divrl(pt2_g, l, rgd.r_g[:maxlen])
        spline = rgd.spline(pt2_g, rgd.r_g[maxlen - 1], l=l)
        pt_j.append(spline)
    return pt_j


# XXX not used at the moment
def local_potential_to_spline(rgd, vbar_g, filter=None):
    vbar_g = vbar_g.copy()
    rcut = rgd.r_g[len(vbar_g) - 1]
    if filter is not None:
        filter(rgd, rcut, vbar_g, l=0)
    # vbar = Spline.from_data(0, rcut, vbar_g)
    vbar = rgd.spline(vbar_g, rgd.r_g[len(vbar_g) - 1], l=0)
    return vbar


def get_radial_hartree_energy(r_g, rho_g):
    """Get energy of l=0 compensation charge on equidistant radial grid."""

    # At least in some cases the zeroth point is moved to 1e-8 or so to
    # prevent division by zero and the like, so:
    dr = r_g[2] - r_g[1]
    rho_r_dr_g = dr * r_g * rho_g
    vh_r_g = np.zeros(len(r_g))  # "r * vhartree"
    hartree_solve(0, rho_r_dr_g, r_g, vh_r_g)
    return 2.0 * np.pi * (rho_r_dr_g * vh_r_g).sum()


def screen_potential(r, v, charge, rcut=None, a=None):
    """Split long-range potential into short-ranged contributions.

    The potential v is a long-ranted potential with the asymptotic form Z/r
    corresponding to the given charge.

    Return a potential vscreened and charge distribution rhocomp such that

      v(r) = vscreened(r) + vHartree[rhocomp](r).

    The returned quantities are truncated to a reasonable cutoff radius.
    """
    vr = v * r + charge

    if rcut is None:
        err = 0.0
        i = len(vr)
        while err < 1e-4:
            # Things can be a bit sensitive to the threshold.  The O.pz-mt
            # setup gets 20-30 Bohr long compensation charges if it's 1e-6.
            i -= 1
            err = abs(vr[i])
        i += 1

        icut = np.searchsorted(r, r[i] * 1.1)
    else:
        icut = np.searchsorted(r, rcut)
    rcut = r[icut]
    rshort = r[:icut].copy()
    if rshort[0] < 1e-16:
        rshort[0] = 1e-10

    if a is None:
        a = rcut / 5.0  # XXX why is this so important?
    vcomp = np.zeros_like(rshort)
    vcomp = charge * erf(rshort / (np.sqrt(2.0) * a)) / rshort
    # XXX divide by r
    rhocomp = charge * (np.sqrt(2.0 * np.pi) * a)**(-3) * \
        np.exp(-0.5 * (rshort / a)**2)
    vscreened = v[:icut] + vcomp
    return vscreened, rhocomp


def figure_out_valence_states(ppdata):
    from my_gpaw25.atom.configurations import configurations
    from ase.data import chemical_symbols
    # ppdata.symbol may not be a chemical symbol so use Z
    chemical_symbol = chemical_symbols[ppdata.Z]
    Z, config = configurations[chemical_symbol]
    assert Z == ppdata.Z

    # Okay, we need to figure out occupations f_ln when we don't know
    # any info about existing states on the pseudopotential.
    #
    # The plan is to loop over all states and count until only the correct
    # number of valence electrons "remain".
    nelectrons = 0
    ncore = ppdata.Z - ppdata.Nv

    energies = [c[3] for c in config]
    args = np.argsort(energies)
    config = list(np.array(config, dtype=object)[args])

    nelectrons = 0
    ncore = ppdata.Z - ppdata.Nv
    assert ppdata.Nv > 0
    iterconfig = iter(config)
    if ncore > 0:
        for n, l, occ, eps in iterconfig:
            nelectrons += occ
            if nelectrons == ncore:
                break
            elif nelectrons >= ncore:
                raise ValueError('Cannot figure out what states should exist '
                                 'on this pseudopotential.')

    f_ln = {}
    l_j = []
    f_j = []
    n_j = []
    for n, l, occ, eps in iterconfig:
        f_ln.setdefault(l, []).append(occ)
        l_j.append(l)
        f_j.append(occ)
        n_j.append(n)
    lmax = max(f_ln.keys())
    f_ln = [f_ln.get(l, []) for l in range(lmax + 1)]
    return n_j, l_j, f_j, f_ln


def generate_basis_functions(ppdata):
    class SimpleBasis(Basis):
        def __init__(self, symbol, l_j, n_j):
            rgd = EquidistantRadialGridDescriptor(0.02, 160)
            Basis.__init__(self, symbol, 'simple', readxml=False, rgd=rgd)
            self.generatordata = 'simple'
            bf_j = self.bf_j
            rcgauss = rgd.r_g[-1] / 3.0
            gauss_g = np.exp(-(rgd.r_g / rcgauss)**2.0)
            for l, n in zip(l_j, n_j):
                phit_g = rgd.r_g**l * gauss_g
                norm = (rgd.integrate(phit_g**2) / (4 * np.pi))**0.5
                phit_g /= norm
                bf = BasisFunction(n, l, rgd.r_g[-1], phit_g, 'gaussian')
                bf_j.append(bf)
    # l_orb_J = [state.l for state in self.data['states']]
    b1 = SimpleBasis(ppdata.symbol, ppdata.l_orb_J, ppdata.n_j)
    apaw = AtomPAW(ppdata.symbol, [ppdata.f_ln], h=0.05, rcut=9.0,
                   basis={ppdata.symbol: b1},
                   setups={ppdata.symbol: ppdata},
                   maxiter=60,
                   txt=None)
    basis = apaw.extract_basis_functions(ppdata.n_j, ppdata.l_j)
    return basis


def pseudoplot(pp, show=True):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    wfsax = fig.add_subplot(221)
    ptax = fig.add_subplot(222)
    vax = fig.add_subplot(223)
    rhoax = fig.add_subplot(224)

    def spline2grid(spline):
        rcut = spline.get_cutoff()
        r = np.linspace(0.0, rcut, 2000)
        return r, spline.map(r)

    for phit in pp.phit_j:
        r, y = spline2grid(phit)
        wfsax.plot(r, y, label='wf l=%d' % phit.get_angular_momentum_number())

    for pt in pp.pt_j:
        r, y = spline2grid(pt)
        ptax.plot(r, y, label='pr l=%d' % pt.get_angular_momentum_number())

    for ghat in pp.ghat_l:
        r, y = spline2grid(ghat)
        rhoax.plot(r, y, label='cc l=%d' % ghat.get_angular_momentum_number())

    r, y = spline2grid(pp.vbar)
    vax.plot(r, y, label='vbar')

    vax.set_ylabel('potential')
    rhoax.set_ylabel('density')
    wfsax.set_ylabel('wfs')
    ptax.set_ylabel('projectors')

    for ax in [vax, rhoax, wfsax, ptax]:
        ax.legend()

    if show:
        plt.show()


class PseudoPotential(BaseSetup):
    is_pseudo = True

    def __init__(self, data, basis=None, filter=None):
        self.data = data

        self.N0_q = None

        self.filename = None
        self.fingerprint = None
        self.symbol = data.symbol
        self.type = data.name

        self.Z = data.Z
        self.Nv = data.Nv
        self.Nc = data.Nc

        self.f_j = data.f_j
        self.n_j = data.n_j
        self.l_j = data.l_j
        self.l_orb_J = data.l_orb_J
        self.nj = len(data.l_j)
        self.nq = self.nj * (self.nj + 1) // 2

        self.ni = sum([2 * l + 1 for l in data.l_j])
        # self.pt_j = projectors_to_splines(data.rgd, data.l_j, data.pt_jg,
        #                                   filter=filter)
        self.pt_j = data.get_projectors()

        if len(self.pt_j) == 0:
            assert False  # not sure yet about the consequences of
            # cleaning this up in the other classes
            self.l_j = [0]
            self.pt_j = [null_spline]

        if basis is None:
            basis = data.create_basis_functions()

        self.basis_functions_J = basis.tosplines()

        # We declare (for the benefit of the wavefunctions reuse method)
        # that we have no PAW projectors as such.  This makes the
        # 'paw' wfs reuse method a no-op.
        self.pseudo_partial_waves_j = []

        self.basis = basis
        self.nao = sum([2 * phit.get_angular_momentum_number() + 1
                        for phit in self.basis_functions_J])

        self.Nct = 0.0
        self.nct = null_spline

        self.lmax = 0

        self.xc_correction = None

        r, l_comp, g_comp = data.get_compensation_charge_functions()
        assert l_comp == [0]  # Presumably only spherical charges
        self.ghat_l = [
            Spline.from_data(l, r[-1], g) for l, g in zip(l_comp, g_comp)
        ]
        self.rcgauss = data.rcgauss

        # accuracy is rather sensitive to this
        # self.vbar = local_potential_to_spline(data.rgd, data.vbar_g,
        #                                      filter=filter)
        self.vbar = data.get_local_potential()
        # XXX HGH and UPF use different radial grids, and this for
        # some reason makes it difficult to use the exact same code to
        # construct vbar and projectors.  This should be fixed since
        # either type of rgd should be able to always produce a valid
        # and equivalent spline transparently.

        _np = self.ni * (self.ni + 1) // 2
        self.Delta0 = data.Delta0
        self.Delta_pL = np.zeros((_np, 1))

        self.E = 0.0
        self.Kc = 0.0
        self.M = -data.Eh_compcharge
        self.M_p = np.zeros(_np)
        self.M_pp = np.zeros((_np, _np))
        self.M_wpp = {}

        self.K_p = data.expand_hamiltonian_matrix()
        self.MB = 0.0
        self.MB_p = np.zeros(_np)
        self.dO_ii = np.zeros((self.ni, self.ni))

        # We don't really care about these variables
        self.rcutfilter = None
        self.rcore = None

        self.N0_p = np.zeros(_np)  # not really implemented
        if hasattr(data, 'nabla_iiv'):
            self.nabla_iiv = data.nabla_iiv
        else:
            self.nabla_iiv = None
        self.rxnabla_iiv = None
        self.phicorehole_g = None
        self.rgd = data.rgd
        self.rcut_j = data.rcut_j
        self.tauct = None
        self.Delta_iiL = np.zeros((self.ni, self.ni, 1))
        self.B_ii = None
        self.dC_ii = None
        self.X_p = None
        self.X_wp = {}
        self.X_pg = None
        self.ExxC = None
        self.ExxC_w = {}
        self.dEH0 = 0.0
        self.dEH_p = np.zeros(_np)
        self.extra_xc_data = {}

        self.wg_lg = None
        self.g_lg = None
        self.local_corr = LocalCorrectionVar(None)
