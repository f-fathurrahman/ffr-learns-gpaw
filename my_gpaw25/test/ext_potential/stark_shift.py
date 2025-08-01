from math import sqrt, pi

import pytest
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from my_gpaw25.mpi import rank, size
from my_gpaw25 import GPAW
from my_gpaw25.external import ConstantElectricField
from my_gpaw25.utilities import packed_index
from my_gpaw25.pair_density import PairDensity

# Three ways to compute the polarizability of hydrogen:
# 1. Perturbation theory
# 2. Constant electric field
# 3. Middle of an electric dipole -- not tested here

# Note: The analytical value for the polarizability
# is 4.5 a0**3 (e.g. PRA 33, 3671), while the experimental
# value is 4.6 a0**3 (e.g. PR 133, A629).


@pytest.mark.skip(reason='too-slow')
def test_stark_shift():
    to_au = Hartree / Bohr**2

    def dipole_op(c, state1, state2, k=0, s=0):
        # Taken from KSSingle, maybe make this accessible in
        # KSSingle?
        wfs = c.wfs
        gd = wfs.gd

        kpt = None
        for i in wfs.kpt_u:
            if i.k == k and i.s == s:
                kpt = i

        pd = PairDensity(c)
        pd.initialize(kpt, state1, state2)

        # coarse grid contribution
        # <i|r|j> is the negative of the dipole moment (because of negative
        # e- charge)
        me = -gd.calculate_dipole_moment(pd.get())

        # augmentation contributions
        ma = np.zeros(me.shape)
        pos_av = c.get_atoms().get_positions() / Bohr
        for a, P_ni in kpt.P_ani.items():
            Ra = pos_av[a]
            Pi_i = P_ni[state1]
            Pj_i = P_ni[state2]
            Delta_pL = wfs.setups[a].Delta_pL
            ni = len(Pi_i)

            ma0 = 0
            ma1 = np.zeros(me.shape)
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
                        ma1 += np.array([
                            Delta_pL[ij, 3], Delta_pL[ij, 1], Delta_pL[ij, 2]
                        ]) * pij
            ma += sqrt(4 * pi / 3) * ma1 + Ra * sqrt(4 * pi) * ma0
        gd.comm.sum(ma)

        me += ma

        return me * Bohr

    # Currently only works on a single processor
    assert size == 1

    maxfield = 0.01
    nfs = 5  # number of field
    nbands = 30  # number of bands
    h = 0.20  # grid spacing

    debug = not False

    if debug:
        txt = 'my_gpaw25.out'
    else:
        txt = None

    test1 = True
    test2 = True

    a0 = 6.0
    a = Atoms('H', positions=[[a0 / 2, a0 / 2, a0 / 2]], cell=[a0, a0, a0])

    alpha1 = None
    alpha2 = None

    # Test 1

    if test1:
        c = GPAW(mode='fd',
                 h=h,
                 nbands=nbands + 10,
                 spinpol=True,
                 hund=True,
                 xc='LDA',
                 eigensolver='cg',
                 convergence={
                     'bands': nbands,
                     'eigenstates': 3.3e-4
                 },
                 maxiter=1000,
                 txt=txt)
        a.calc = c
        a.get_potential_energy()

        o1 = c.get_occupation_numbers(spin=0)

        if o1[0] > 0.0:
            spin = 0
        else:
            spin = 1

        alpha = 0.0

        ev = c.get_eigenvalues(0, spin)
        for i in range(1, nbands):
            mu_x, mu_y, mu_z = dipole_op(c, 0, i, k=0, s=spin)

            alpha += mu_z**2 / (ev[i] - ev[0])

        alpha *= 2

        if rank == 0 and debug:
            print('From perturbation theory:')
            print('  alpha = ', alpha, ' A**2/eV')
            print('  alpha = ', alpha * to_au, ' Bohr**3')

        alpha1 = alpha

    ###

    c = GPAW(
        mode='fd',
        h=h,
        nbands=2,
        spinpol=True,
        hund=True,
        xc='LDA',
        # eigensolver  = 'cg',
        convergence={
            'bands': nbands,
            'eigenstates': 3.3e-4
        },
        maxiter=1000,
        txt=txt)
    a.calc = c

    # Test 2

    if test2:
        e = []
        e1s = []
        d = []
        fields = np.linspace(-maxfield, maxfield, nfs)
        for field in fields:
            if rank == 0 and debug:
                print(field)
            c = c.new(external=ConstantElectricField(field))
            a.calc = c
            etot = a.get_potential_energy()
            e += [etot]
            ev0 = c.get_eigenvalues(0)
            ev1 = c.get_eigenvalues(0, 1)
            e1s += [min(ev0[0], ev1[0])]
            dip = c.get_dipole_moment()
            d += [dip[2]]

        pol1, dummy = np.polyfit(fields, d, 1)
        pol2, dummy1, dummy2 = np.polyfit(fields, e1s, 2)

        if rank == 0 and debug:
            print('From shift in 1s-state at constant electric field:')
            print('  alpha = ', -pol2, ' A**2/eV')
            print('  alpha = ', -pol2 * to_au, ' Bohr**3')

            print('From dipole moment at constant electric field:')
            print('  alpha = ', pol1, ' A**2/eV')
            print('  alpha = ', pol1 * to_au, ' Bohr**3')

            np.savetxt('ecf.out', np.transpose([fields, e, e1s, d]))

        assert abs(pol1 + pol2) < 0.002
        alpha2 = (pol1 - pol2) / 2

    # # This is a very, very rough test
    assert alpha1 == pytest.approx(alpha2, abs=0.01)
