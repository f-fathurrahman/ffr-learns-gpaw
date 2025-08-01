from math import pi, sqrt

import numpy as np
from ase.atoms import Atoms
from scipy.linalg import eigh

from my_gpaw25.calculator import GPAW
from my_gpaw25.wavefunctions.base import WaveFunctions
from my_gpaw25.atom.radialgd import EquidistantRadialGridDescriptor
from my_gpaw25.utilities import unpack_hermitian
from my_gpaw25.occupations import OccupationNumberCalculator
import my_gpaw25.mpi as mpi


class MakeWaveFunctions:
    name = 'atompaw'
    interpolation = 9
    force_complex_dtype = False

    def __init__(self, gd):
        self.gd = gd

    def __call__(self, paw, gd, *args, **kwargs):
        return AtomWaveFunctions(self.gd, *args, **kwargs)


class AtomWaveFunctionsArray:
    def __init__(self, psit_nG):
        self.array = psit_nG


class AtomWaveFunctions(WaveFunctions):
    mode = 'atompaw'

    def initialize(self, density, hamiltonian, spos_ac):
        setup = self.setups[0]
        bf = AtomBasisFunctions(self.gd, setup.basis_functions_J)
        density.initialize_from_atomic_densities(bf)
        hamiltonian.update(density)
        return 0, 0

    def add_to_density_from_k_point(self, nt_sG, kpt):
        nt_sG[kpt.s] += np.dot(kpt.f_n / 4 / pi, kpt.psit_nG**2)

    def summary(self, log):
        log('Mode: Spherically symmetric atomic solver')


class AtomPoissonSolver:
    def get_description(self):
        return 'Radial equidistant'

    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.relax_method = 0
        self.nn = 1

    def initialize(self):
        pass

    def get_stencil(self):
        return 'Exact'

    def solve(self, vHt_g, rhot_g, charge=0, timer=None):
        r = self.gd.r_g
        dp = rhot_g * r * self.gd.dr_g
        dq = dp * r
        p = np.add.accumulate(dp[::-1])[::-1]
        q = np.add.accumulate(dq[::-1])[::-1]
        vHt_g[:] = 4 * pi * (p - 0.5 * dp - (q - 0.5 * dq - q[0]) / r)
        return 1


class AtomEigensolver:
    def __init__(self, gd, f_sln):
        self.gd = gd
        self.f_sln = f_sln
        self.error = 0.0
        self.initialized = False

    def reset(self):
        self.initialized = False

    def initialize(self, wfs):
        r = self.gd.r_g
        h = r[0]
        N = len(r)
        lmax = len(self.f_sln[0]) - 1

        self.T_l = [np.eye(N) * (1.0 / h**2)]
        self.T_l[0].flat[1::N + 1] = -0.5 / h**2
        self.T_l[0].flat[N::N + 1] = -0.5 / h**2
        for l in range(1, lmax + 1):
            self.T_l.append(self.T_l[0] + np.diag(l * (l + 1) / 2.0 / r**2))

        self.S_l = [np.eye(N) for l in range(lmax + 1)]
        setup = wfs.setups[0]
        self.pt_j = np.array([[pt(x) * x**l for x in r]
                              for pt, l in zip(setup.pt_j, setup.l_j)])

        dS_ii = setup.dO_ii
        i1 = 0
        for pt1, l1 in zip(self.pt_j, setup.l_j):
            i2 = 0
            for pt2, l2 in zip(self.pt_j, setup.l_j):
                if l1 == l2 and l1 <= lmax:
                    self.S_l[l1] += (np.outer(pt1 * r, pt2 * r) *
                                     h * dS_ii[i1, i2])
                i2 += 2 * l2 + 1
            i1 += 2 * l1 + 1

        for kpt in wfs.kpt_u:
            kpt.eps_n = np.empty(wfs.bd.nbands)
            kpt.psit = AtomWaveFunctionsArray(self.gd.empty(wfs.bd.nbands))
            kpt.projections = {0: np.zeros((wfs.bd.nbands, len(dS_ii)))}

        self.initialized = True

    def iterate(self, hamiltonian, wfs):
        if not self.initialized:
            self.initialize(wfs)

        r = self.gd.r_g
        h = r[0]
        N = len(r)
        lmax = len(self.f_sln[0]) - 1
        setup = wfs.setups[0]

        e_n = np.zeros(N)

        for s in range(wfs.nspins):
            dH_ii = unpack_hermitian(hamiltonian.dH_asp[0][s])
            kpt = wfs.kpt_u[s]
            N1 = 0
            for l in range(lmax + 1):
                H = self.T_l[l] + np.diag(hamiltonian.vt_sg[s])
                i1 = 0
                for pt1, l1 in zip(self.pt_j, setup.l_j):
                    i2 = 0
                    for pt2, l2 in zip(self.pt_j, setup.l_j):
                        if l1 == l2 == l:
                            H += (h * dH_ii[i1, i2] *
                                  np.outer(pt1 * r, pt2 * r))
                        i2 += 2 * l2 + 1
                    i1 += 2 * l1 + 1
                e_n, H = eigh(H, self.S_l[l].copy())

                for n in range(len(self.f_sln[s][l])):
                    N2 = N1 + 2 * l + 1
                    kpt.eps_n[N1:N2] = e_n[n]
                    kpt.psit_nG[N1:N2] = H[:, n] / r / sqrt(h)
                    i1 = 0
                    for pt, ll in zip(self.pt_j, setup.l_j):
                        i2 = i1 + 2 * ll + 1
                        if ll == l:
                            P = np.dot(kpt.psit_nG[N1], pt * r**2) * h
                            kpt.P_ani[0][N1:N2, i1:i2] = P * np.eye(2 * l + 1)
                        i1 = i2
                    N1 = N2


class AtomLocalizedFunctionsCollection:
    def __init__(self, gd, spline_aj):
        self.gd = gd
        spline = spline_aj[0][0]
        self.b_g = np.array([spline(r) for r in gd.r_g]) / sqrt(4 * pi)
        self.nfunctions = sum(2 * spline.get_angular_momentum_number() + 1
                              for spline in spline_aj[0])

    def set_positions(self, spos_ac, atom_partition=None):
        pass

    def add(self, a_xG, c_axi=1.0, q=-1):
        assert q == -1
        if isinstance(c_axi, float):
            a_xG += c_axi * self.b_g
        else:
            a_xG += c_axi[0][0] * self.b_g

    def integrate(self, a_g, c_ai, q=-1):
        assert a_g.ndim == 1
        assert q == -1
        c_ai[0][0] = self.gd.integrate(a_g, self.b_g)
        c_ai[0][1:] = 0.0

    def dict(self):
        return {0: np.empty(self.nfunctions)}


class AtomBasisFunctions:
    def __init__(self, gd, bfs_J):
        self.gd = gd
        self.bl_J = []
        self.Mmax = 0

        for bf in bfs_J:
            l = bf.get_angular_momentum_number()
            self.bl_J.append((np.array([bf(x) * x**l for x in gd.r_g]), l))
            self.Mmax += 2 * l + 1
        self.atom_indices = [0]
        self.my_atom_indices = [0]

    def set_positions(self, spos_ac):
        pass

    def add_to_density(self, nt_sG, f_asi):
        i = 0
        for b_g, l in self.bl_J:
            nt_sG += f_asi[0][:, i:i + 1] * (2 * l + 1) / 4 / pi * b_g**2
            i += 2 * l + 1


class AtomGridDescriptor(EquidistantRadialGridDescriptor):
    def __init__(self, h, rcut):
        ng = int(float(rcut) / h + 0.5) - 1
        rcut = ng * h
        EquidistantRadialGridDescriptor.__init__(self, h, ng, h0=h)
        self.sdisp_cd = np.empty((3, 2))
        self.comm = mpi.serial_comm
        self.pbc_c = np.zeros(3, bool)
        self.cell_cv = np.eye(3) * rcut
        self.N_c = np.ones(3, dtype=int) * 2 * ng
        self.h_cv = self.cell_cv / self.N_c
        self.dv = (rcut / 2 / ng)**3
        self.orthogonal = False
        self.parsize_c = (1, 1, 1)

    def get_ranks_from_positions(self, spos_ac):
        return np.array([0])

    def refine(self):
        return self

    def get_lfc(self, gd, spline_aj):
        return AtomLocalizedFunctionsCollection(gd, spline_aj)

    def integrate(self, a_xg, b_xg=None, global_integral=True):
        """Integrate function(s) in array over domain."""
        if b_xg is None:
            return np.dot(a_xg, self.dv_g)
        else:
            return np.dot(a_xg * b_xg, self.dv_g)

    def calculate_dipole_moment(self, rhot_g):
        return np.zeros(3)

    def symmetrize(self, a_g, op_scc, ft_sc=None):
        pass

    def get_grid_spacings(self):
        return self.h_cv.diagonal()

    def get_size_of_global_array(self, pad=None):
        return np.array(len(self.N_c))

    def new_descriptor(self, *args, **kwargs):
        return self

    def get_processor_position_from_rank(self, rank):
        return (0, 0, 0)


class AtomOccupations(OccupationNumberCalculator):
    extrapolate_factor = 0.0

    def __init__(self, f_sln):
        self.f_sln = f_sln
        OccupationNumberCalculator.__init__(self)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess,
                   fix_fermi_level=False):
        for s, f_n in enumerate(f_qn):
            n1 = 0
            for l, f0_n in enumerate(self.f_sln[s]):
                for f in f0_n:
                    n2 = n1 + 2 * l + 1
                    f_n[n1:n2] = f / (2 * l + 1) / 2
                    n1 = n2

        return np.inf, 0.0


class AtomPAW(GPAW):
    def __init__(self, symbol, f_sln, h=0.05, rcut=10.0, **kwargs):
        assert len(f_sln) in [1, 2]
        self.symbol = symbol

        gd = AtomGridDescriptor(h, rcut)
        GPAW.__init__(self,
                      mode=MakeWaveFunctions(gd),
                      eigensolver=AtomEigensolver(gd, f_sln),
                      poissonsolver=AtomPoissonSolver(),
                      nbands=sum([(2 * l + 1) * len(f_n)
                                  for l, f_n in enumerate(f_sln[0])]),
                      communicator=mpi.serial_comm,
                      parallel=dict(augment_grids=False),
                      occupations=AtomOccupations(f_sln),
                      **kwargs)
        # Initialize function will raise an error unless we set a (bogus) cell
        self.initialize(Atoms(symbol, calculator=self,
                              cell=np.eye(3)))
        self.density.charge_eps = 1e-3
        self.calculate(system_changes=['positions'])

    def dry_run(self):
        pass

    def state_iter(self):
        """Yield the tuples (l, n, f, eps, psit_G) of states.

        Skips degenerate states."""
        f_sln = self.wfs.occupations.f_sln
        assert len(f_sln) == 1, 'Not yet implemented with more spins'
        f_ln = f_sln[0]
        kpt = self.wfs.kpt_u[0]

        band = 0
        for l, f_n in enumerate(f_ln):
            for n, f in enumerate(f_n):
                psit_G = kpt.psit_nG[band]
                eps = kpt.eps_n[band]
                yield l, n, f, eps, psit_G
                band += 2 * l + 1

    def extract_basis_functions(self, n_j, l_j, basis_name='atompaw.sz'):
        """Create BasisFunctions object with pseudo wave functions."""
        from my_gpaw25.basis_data import Basis, BasisFunction
        assert self.wfs.nspins == 1

        d = self.wfs.gd.r_g[0]
        ng = self.wfs.gd.N + 1
        rgd = EquidistantRadialGridDescriptor(d, ng)
        basis = Basis(self.symbol, basis_name, readxml=False, rgd=rgd)
        basis.generatorattrs = {}  # attrs of the setup maybe
        basis.generatordata = 'AtomPAW'  # version info too?

        bf_j = basis.bf_j
        for l, n, f, eps, psit_G in self.state_iter():
            n = [N for N, L in zip(n_j, l_j) if L == l][n]
            bf_g = rgd.empty()
            bf_g[0] = 0.0
            bf_g[1:] = psit_G
            bf_g *= np.sign(psit_G[-1])

            # If there's no node at zero, we shouldn't set bf_g to zero
            # We'll make an ugly hack
            if abs(bf_g[1]) > 3.0 * abs(bf_g[2] - bf_g[1]):
                bf_g[0] = bf_g[1]
            bf = BasisFunction(n, l, self.wfs.gd.r_g[-1], bf_g,
                               f'{n}{"spdfgh"[l]} e={eps:.3f} f={f:.3f}')
            bf_j.append(bf)
        return basis
