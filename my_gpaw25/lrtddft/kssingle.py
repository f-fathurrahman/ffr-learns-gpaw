"""Kohn-Sham single particle excitations realated objects.

"""
import sys
import json
import numpy as np
from copy import copy

from ase.units import Bohr, Hartree, alpha

import my_gpaw25.mpi as mpi
from my_gpaw25.utilities import packed_index
from my_gpaw25.lrtddft.excitation import Excitation, ExcitationList, get_filehandle
from my_gpaw25.pair_density import PairDensity
from my_gpaw25.fd_operators import Gradient
from my_gpaw25.utilities.tools import coordinates
from .kssrestrictor import KSSRestrictor


class KSSingles(ExcitationList):
    """Kohn-Sham single particle excitations

    Input parameters:

    calculator:
      the calculator object after a ground state calculation

    nspins:
      number of spins considered in the calculation
      Note: Valid only for unpolarised ground state calculation

    eps:
      Minimal occupation difference for a transition (default 0.001)

    istart:
      First occupied state to consider
    jend:
      Last unoccupied state to consider
    energy_range:
      The energy range [emin, emax] or emax for KS transitions to use as basis
    """
    def __init__(self,
                 restrict={},
                 log=None,
                 txt=None):
        ExcitationList.__init__(self, log=log, txt=txt)
        self.world = mpi.world

        self.restrict = KSSRestrictor()
        self.restrict.update(restrict)

    def calculate(self, atoms, nspins=None):
        calculator = atoms.calc
        self.calculator = calculator

        # LCAO calculation requires special actions
        self.lcao = calculator.wfs.mode == 'lcao'

        # deny hybrids as their empty states are wrong
#        gsxc = calculator.hamiltonian.xc
#        hybrid = hasattr(gsxc, 'hybrid') and gsxc.hybrid > 0.0
#        assert(not hybrid)

        # ensure correctly initialized wave functions
        calculator.converge_wave_functions()
        self.world = calculator.wfs.world

        # parallelization over bands not yet supported
        assert calculator.wfs.bd.comm.size == 1

        # do the evaluation
        self.select(nspins)

        trkm = self.get_trk()
        self.log('KSS {} transitions (restrict={})'.format(
            len(self), self.restrict))
        self.log('KSS TRK sum %g (%g,%g,%g)' %
                 (np.sum(trkm) / 3., trkm[0], trkm[1], trkm[2]))
        pol = self.get_polarizabilities(lmax=3)
        self.log('KSS polarisabilities(l=0-3) %g, %g, %g, %g' %
                 tuple(pol.tolist()))
        return self

    @staticmethod
    def emin_emax(energy_range):
        emin = -sys.float_info.max
        emax = sys.float_info.max
        if energy_range is not None:
            try:
                emin, emax = energy_range
                emin /= Hartree
                emax /= Hartree
            except TypeError:
                emax = energy_range / Hartree
        return emin, emax

    def select(self, nspins=None):
        """Select KSSingles according to the given criterium."""

        # criteria
        emin, emax = self.restrict.emin_emax()
        istart = self.restrict['istart']
        jend = self.restrict['jend']
        eps = self.restrict['eps']

        if not hasattr(self, 'calculator'):  # I'm read from a file
            # throw away all not needed entries
            for i, ks in reversed(list(enumerate(self))):
                if not self.restrict.is_good(ks):
                    del self[i]
            return None

        paw = self.calculator
        wfs = paw.wfs
        self.dtype = wfs.dtype
        self.kpt_u = wfs.kpt_u

        if not self.lcao and self.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions in calculator!')

        # here, we need to take care of the spins also for
        # closed shell systems (Sz=0)
        # vspin is the virtual spin of the wave functions,
        #       i.e. the spin used in the ground state calculation
        # pspin is the physical spin of the wave functions
        #       i.e. the spin of the excited states
        self.nvspins = wfs.nspins
        self.npspins = wfs.nspins
        fijscale = 1
        ispins = [0]
        nks = wfs.kd.nibzkpts * wfs.kd.nspins
        if self.nvspins < 2:
            if (nspins or 0) > self.nvspins:
                self.npspins = nspins
                fijscale = 0.5
                ispins = [0, 1]
                nks *= 2

        kpt_comm = self.calculator.wfs.kd.comm
        nbands = len(self.kpt_u[0].f_n)

        # select
        take = np.zeros((nks, nbands, nbands), dtype=int)
        u = 0
        for ispin in ispins:
            for k in range(wfs.kd.nibzkpts):
                q = k - wfs.kd.k0
                for s in range(wfs.nspins):
                    if q >= 0 and q < wfs.kd.mynk:
                        kpt = wfs.kpt_qs[q][s]
                        for i in range(nbands):
                            for j in range(i + 1, nbands):
                                fij = (kpt.f_n[i] - kpt.f_n[j]) / kpt.weight
                                epsij = kpt.eps_n[j] - kpt.eps_n[i]
                                if (fij > eps and
                                    epsij >= emin and epsij < emax and
                                        i >= istart and j <= jend):
                                    take[u, i, j] = 1
                    u += 1
        kpt_comm.sum(take)

        self.log()
        self.log('Kohn-Sham single transitions')
        self.log()

        # calculate in parallel
        u = 0
        for ispin in ispins:
            for k in range(wfs.kd.nibzkpts):
                q = k - wfs.kd.k0
                for s in range(wfs.kd.nspins):
                    for i in range(nbands):
                        for j in range(i + 1, nbands):
                            if take[u, i, j]:
                                if q >= 0 and q < wfs.kd.mynk:
                                    kpt = wfs.kpt_qs[q][s]
                                    pspin = max(kpt.s, ispin)
                                    self.append(
                                        KSSingle(i, j, pspin, kpt, paw,
                                                 fijscale=fijscale,
                                                 dtype=self.dtype))
                                else:
                                    self.append(KSSingle(i, j, pspin=0,
                                                         kpt=None, paw=paw,
                                                         dtype=self.dtype))
                    u += 1

        # distribute
        for kss in self:
            kss.distribute()

    @classmethod
    def read(cls, filename=None, fh=None, restrict={}, log=None):
        """Read myself from a file"""
        assert (filename is not None) or (fh is not None)

        def fail(f):
            raise RuntimeError(f.name + ' does not contain ' +
                               cls.__class__.__name__ + ' data')
        if fh is None:
            f = get_filehandle(cls, filename)

            # there can be other information, i.e. the LrTDDFT header
            try:
                content = f.read()
                f.seek(content.index('# KSSingles'))
                del content
                f.readline()
            except ValueError:
                fail(f)
        else:
            f = fh
            # we assume to be at the right place and read the header
            if not f.readline().strip() == '# KSSingles':
                fail(f)

        words = f.readline().split()
        n = int(words[0])
        kssl = cls(log=log)
        if len(words) == 1:
            # very old output style for real wave functions (finite systems)
            kssl.dtype = float
            restrict_from_file = {}
        else:
            if words[1].startswith('complex'):
                kssl.dtype = complex
            else:
                kssl.dtype = float
            restrict_from_file = json.loads(f.readline())
            if not isinstance(restrict_from_file, dict):  # old output style
                restrict_from_file = {'eps': restrict_from_file}
        kssl.npspins = 1
        for i in range(n):
            kss = KSSingle(string=f.readline(), dtype=kssl.dtype)
            kssl.append(kss)
            kssl.npspins = max(kssl.npspins, kss.pspin + 1)

        if fh is None:
            f.close()

        kssl.update()
        kssl.restrict.update(restrict_from_file)
        if len(restrict):
            kssl.restrict.update(restrict)
            kssl.select()

        return kssl

    def update(self):
        istart = self[0].i
        jend = 0
        npspins = 1
        nvspins = 1
        for kss in self:
            istart = min(kss.i, istart)
            jend = max(kss.j, jend)
            if kss.pspin == 1:
                npspins = 2
            if kss.spin == 1:
                nvspins = 2
        self.restrict.update({'istart': istart, 'jend': jend})
        self.npspins = npspins
        self.nvspins = nvspins

        if hasattr(self, 'energies'):
            del self.energies

    def set_arrays(self):
        if hasattr(self, 'energies'):
            return
        energies = []
        fij = []
        me = []
        mur = []
        muv = []
        magn = []
        for k in self:
            energies.append(k.energy)
            fij.append(k.fij)
            me.append(k.me)
            mur.append(k.mur)
            if k.muv is not None:
                muv.append(k.muv)
            if k.magn is not None:
                magn.append(k.magn)
        self.energies = np.array(energies)
        self.fij = np.array(fij)
        self.me = np.array(me)
        self.mur = np.array(mur)
        if len(muv):
            self.muv = np.array(muv)
        else:
            self.muv = None
        if len(magn):
            self.magn = np.array(magn)
        else:
            self.magn = None

    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files.
        """
        if self.world.rank != 0:
            return

        if fh is None:
            f = get_filehandle(self, filename, mode='w')
        else:
            f = fh

        f.write('# KSSingles\n')
        f.write(f'{len(self)} {np.dtype(self.dtype)}\n')
        f.write(json.dumps(self.restrict.values) + '\n')
        for kss in self:
            f.write(kss.outstring())
        if fh is None:
            f.close()

    def overlap(self, ov_nn, other):
        """Matrix element overlaps determined from wave function overlaps.

        Parameters
        ----------
        ov_nn: array
            Wave function overlap factors from a displaced calculator.
            Index 0 corresponds to our own wavefunctions conjugated and
            index 1 to the others' wavefunctions

        Returns
        -------
        ov_pp: array
            Overlap corresponding to matrix elements.
            Index 0 corresponds to our own matrix elements conjugated and
            index 1 to the others' matrix elements
        """
        n0 = len(self)
        n1 = len(other)
        ov_pp = np.zeros((n0, n1), dtype=ov_nn.dtype)
        i1_p = [ex.i for ex in other]
        j1_p = [ex.j for ex in other]
        for p0, ex0 in enumerate(self):
            ov_pp[p0, :] = ov_nn[ex0.i, i1_p].conj() * ov_nn[ex0.j, j1_p]
        return ov_pp


class KSSingle(Excitation, PairDensity):
    """Single Kohn-Sham transition containing all its indices

      pspin=physical spin
      spin=virtual  spin, i.e. spin in the ground state calc.
      kpt=the Kpoint object
      fijscale=weight for the occupation difference::
      me  = sqrt(fij*epsij) * <i|r|j>
      mur = - <i|r|a>
      muv = - <i|nabla|a>/omega_ia with omega_ia>0
      magn = <i|[r x nabla]|a> / (2 m_e c)
    """

    def __init__(self, iidx=None, jidx=None, pspin=None, kpt=None,
                 paw=None, string=None, fijscale=1, dtype=float):
        """
        iidx: index of occupied state
        jidx: index of empty state
        pspin: physical spin
        kpt: kpoint object,
        paw: calculator,
        string: string to be initialized from
        fijscale:
        dtype: dtype of matrix elements
        """
        if string is not None:
            self.fromstring(string, dtype)
            return None

        # normal entry

        PairDensity.__init__(self, paw)
        PairDensity.initialize(self, kpt, iidx, jidx)

        self.pspin = pspin

        self.energy = 0.0
        self.fij = 0.0

        self.me = np.zeros((3), dtype=dtype)
        self.mur = np.zeros((3), dtype=dtype)
        self.muv = np.zeros((3), dtype=dtype)
        self.magn = np.zeros((3), dtype=dtype)

        self.kpt_comm = paw.wfs.kd.comm

        # leave empty if not my kpt
        if kpt is None:
            return

        wfs = paw.wfs
        gd = wfs.gd

        self.energy = kpt.eps_n[jidx] - kpt.eps_n[iidx]
        self.fij = (kpt.f_n[iidx] - kpt.f_n[jidx]) * fijscale

        # calculate matrix elements -----------

        # length form ..........................

        # course grid contribution
        # <i|r|j> is the negative of the dipole moment (because of negative
        # e- charge)
        me = - gd.calculate_dipole_moment(self.get())

        # augmentation contributions
        ma = np.zeros(me.shape, dtype=dtype)
        pos_av = paw.atoms.get_positions() / Bohr
        for a, P_ni in kpt.P_ani.items():
            Ra = pos_av[a]
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            Delta_pL = wfs.setups[a].Delta_pL
            ni = len(Pi_i)
            ma0 = 0
            ma1 = np.zeros(me.shape, dtype=me.dtype)
            for i in range(ni):
                for j in range(ni):
                    pij = Pi_i[i] * Pj_i[j]
                    ij = packed_index(i, j, ni)
                    # L=0 term
                    ma0 += Delta_pL[ij, 0] * pij
                    # L=1 terms
                    if wfs.setups[a].lmax >= 1:
                        # see spherical_harmonics.py for
                        # L=1:y L=2:z; L=3:x
                        ma1 += np.array([Delta_pL[ij, 3], Delta_pL[ij, 1],
                                         Delta_pL[ij, 2]]) * pij
            ma += np.sqrt(4 * np.pi / 3) * ma1 + Ra * np.sqrt(4 * np.pi) * ma0
        gd.comm.sum(ma)

        self.me = np.sqrt(self.energy * self.fij) * (me + ma)
        self.mur = - (me + ma)

        # velocity form .............................

        if self.lcao:
            self.wfi = _get_and_distribute_wf(wfs, iidx, kpt.k, pspin)
            self.wfj = _get_and_distribute_wf(wfs, jidx, kpt.k, pspin)

        me = np.zeros(self.mur.shape, dtype=dtype)

        # get derivatives
        dtype = self.wfj.dtype
        dwfj_cg = gd.empty((3), dtype=dtype)
        if not hasattr(gd, 'ddr'):
            gd.ddr = [Gradient(gd, c, dtype=dtype, n=2).apply
                      for c in range(3)]
        for c in range(3):
            gd.ddr[c](self.wfj, dwfj_cg[c], kpt.phase_cd)
            me[c] = gd.integrate(self.wfi.conj() * dwfj_cg[c])

        # XXX is this the best choice, maybe center of mass?
        origin = 0.5 * np.diag(paw.wfs.gd.cell_cv)

        # augmentation contributions

        # <psi_i|grad|psi_j>
        ma = np.zeros(me.shape, dtype=me.dtype)
        # Ra x <psi_i|grad|psi_j> for magnetic transition dipole
        mRa = np.zeros(me.shape, dtype=me.dtype)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            nabla_iiv = paw.wfs.setups[a].nabla_iiv
            ma_c = np.zeros(me.shape, dtype=me.dtype)
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma_c[c] += Pi * Pj * nabla_iiv[i1, i2, c]
            mRa += np.cross(paw.atoms[a].position / Bohr - origin, ma_c)
            ma += ma_c
        gd.comm.sum(ma)
        gd.comm.sum(mRa)

        self.muv = - (me + ma) / self.energy

        # magnetic transition dipole ................

        # m_ij = -(1/2c) <i|L|j> = i/2c <i|r x p|j>
        # see Autschbach et al., J. Chem. Phys., 116, 6930 (2002)

        r_cg, r2_g = coordinates(gd, origin=origin)
        magn = np.zeros(me.shape, dtype=dtype)

        # <psi_i|r x grad|psi_j>
        wfi_g = self.wfi.conj()
        for ci in range(3):
            cj = (ci + 1) % 3
            ck = (ci + 2) % 3
            magn[ci] = gd.integrate(wfi_g * r_cg[cj] * dwfj_cg[ck] -
                                    wfi_g * r_cg[ck] * dwfj_cg[cj])

        # augmentation contributions
        # <psi_i| r x nabla |psi_j>
        # = <psi_i| (r - Ra + Ra) x nabla |psi_j>
        # = <psi_i| (r - Ra) x nabla |psi_j> + Ra x <psi_i| nabla |psi_j>

        ma = np.zeros(magn.shape, dtype=magn.dtype)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i].conj()
            Pj_i = P_ni[self.j]
            rxnabla_iiv = paw.wfs.setups[a].rxnabla_iiv
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma[c] += Pi * Pj * rxnabla_iiv[i1, i2, c]
        gd.comm.sum(ma)

        self.magn = alpha / 2. * (magn + ma + mRa)

    def distribute(self):
        """Distribute results to all cores."""
        self.spin = self.kpt_comm.sum_scalar(self.spin)
        self.pspin = self.kpt_comm.sum_scalar(self.pspin)
        self.k = self.kpt_comm.sum_scalar(self.k)
        self.weight = self.kpt_comm.sum_scalar(self.weight)
        self.energy = self.kpt_comm.sum_scalar(self.energy)
        self.fij = self.kpt_comm.sum_scalar(self.fij)

        self.kpt_comm.sum(self.me)
        self.kpt_comm.sum(self.mur)
        self.kpt_comm.sum(self.muv)
        self.kpt_comm.sum(self.magn)

    def __add__(self, other):
        """Add two KSSingles"""
        result = copy(self)
        result.me = self.me + other.me
        result.mur = self.mur + other.mur
        result.muv = self.muv + other.muv
        result.magn = self.magn + other.magn
        return result

    def __sub__(self, other):
        """Subtract two KSSingles"""
        result = copy(self)
        result.me = self.me - other.me
        result.mur = self.mur - other.mur
        result.muv = self.muv - other.muv
        result.magn = self.magn - other.magn
        return result

    def __rmul__(self, x):
        return self.__mul__(x)

    def __mul__(self, x):
        """Multiply a KSSingle with a number"""
        assert isinstance(x, (float, int))
        result = copy(self)
        result.me = self.me * x
        result.mur = self.mur * x
        result.muv = self.muv * x
        result.magn = self.magn * x
        return result

    def __truediv__(self, x):
        return self.__mul__(1. / x)

    __div__ = __truediv__

    def fromstring(self, string, dtype=float):
        l = string.split()
        self.i = int(l.pop(0))
        self.j = int(l.pop(0))
        self.pspin = int(l.pop(0))
        self.spin = int(l.pop(0))
        if dtype == float:
            self.k = 0
            self.weight = 1
        else:
            self.k = int(l.pop(0))
            self.weight = float(l.pop(0))
        self.energy = float(l.pop(0))
        self.fij = float(l.pop(0))
        self.mur = np.array([dtype(l.pop(0)) for i in range(3)])
        self.me = - self.mur * np.sqrt(self.energy * self.fij)
        self.muv = self.magn = None
        if len(l):
            self.muv = np.array([dtype(l.pop(0)) for i in range(3)])
        if len(l):
            self.magn = np.array([dtype(l.pop(0)) for i in range(3)])
        return None

    def outstring(self):
        if self.mur.dtype == float:
            string = '{:d} {:d}  {:d} {:d}  {:.10g} {:f}'.format(
                self.i, self.j, self.pspin, self.spin, self.energy, self.fij)
        else:
            string = (
                '{:d} {:d}  {:d} {:d} {:d} {:.10g}  {:g} {:g}'.format(
                    self.i, self.j, self.pspin, self.spin, self.k,
                    self.weight, self.energy, self.fij))
        string += '  '

        def format_me(me):
            string = ''
            if me.dtype == float:
                for m in me:
                    string += f' {m:.5e}'
            else:
                for m in me:
                    string += ' {0.real:.5e}{0.imag:+.5e}j'.format(m)
            return string

        string += '  ' + format_me(self.mur)
        if self.muv is not None:
            string += '  ' + format_me(self.muv)
        if self.magn is not None:
            string += '  ' + format_me(self.magn)
        string += '\n'

        return string

    def __str__(self):
        string = '# <KSSingle> %d->%d %d(%d) eji=%g[eV]' % \
            (self.i, self.j, self.pspin, self.spin,
             self.energy * Hartree)
        if self.me.dtype == float:
            string += f' ({self.me[0]:g},{self.me[1]:g},{self.me[2]:g})'
        else:
            string += f' kpt={self.k:d} w={self.weight:g}'
            string += ' ('
            # use velocity form
            s = - np.sqrt(self.energy * self.fij)
            for c, m in enumerate(s * self.me):
                string += '{0.real:.5e}{0.imag:+.5e}j'.format(m)
                if c < 2:
                    string += ','
            string += ')'
        return string

    def __eq__(self, other):
        """KSSingles are considred equal when their indices are equal."""
        return (self.pspin == other.pspin and self.k == other.k and
                self.i == other.i and self.j == other.j)

    def __hash__(self):
        """Hash similar to __eq__"""
        if not hasattr(self, 'hash'):
            self.hash = hash((self.spin, self.k, self.i, self.j))
        return self.hash

    #
    # User interface: ##
    #

    def get_weight(self):
        return self.fij


def _get_and_distribute_wf(wfs, n, k, s):
    gd = wfs.gd
    wf = wfs.get_wave_function_array(n=n, k=k, s=s, realspace=True,
                                     periodic=False)
    if wfs.world.rank != 0:
        wf = gd.empty(dtype=wfs.dtype, global_array=True)
    wf = np.ascontiguousarray(wf)
    wfs.world.broadcast(wf, 0)
    wfd = gd.empty(dtype=wfs.dtype, global_array=False)
    wfd = gd.distribute(wf)
    return wfd
