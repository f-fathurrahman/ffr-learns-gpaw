from math import pi

from ase import Atoms
from ase.parallel import parprint

from my_gpaw25 import GPAW
import pytest
from my_gpaw25.analyse.expandyl import AngularIntegral, ExpandYl


def test_radial_ylexpand(in_tmp_dir):
    fname = 'H2.gpw'
    donot = ''
    donot = 'donot'
    try:
        calc = GPAW(fname + donot, txt=None)
        H2 = calc.get_atoms()
        calc.converge_wave_functions()
    except FileNotFoundError:
        R = 0.7  # approx. experimental bond length
        a = 2.
        c = 3.
        H2 = Atoms('HH',
                   [(a / 2, a / 2, (c - R) / 2),
                    (a / 2, a / 2, (c + R) / 2)],
                   cell=(a, a, c),
                   pbc=True)
        calc = GPAW(mode='fd', gpts=(12, 12, 16), nbands=2, kpts=(1, 1, 2),
                    convergence={'eigenstates': 1.e-6},
                    txt=None,
                    )
        H2.calc = calc
        H2.get_potential_energy()
        if not donot:
            calc.write(fname)

    # Check that a / h = 10 is rounded up to 12 as always:
    assert (calc.wfs.gd.N_c == (12, 12, 16)).all()

    # AngularIntegral

    gd = calc.density.gd
    ai = AngularIntegral(H2.positions.mean(0), calc.wfs.gd, Rmax=1.5)
    unity_g = gd.zeros() + 1.
    average_R = ai.average(unity_g)
    integral_R = ai.integrate(unity_g)
    for V, average, integral, R, Rm in zip(ai.V_R, average_R, integral_R,
                                           ai.radii(), ai.radii('mean')):
        if V > 0:
            assert average == pytest.approx(1, abs=1.e-9)
            assert integral / (4 * pi * Rm**2) == pytest.approx(1, abs=0.61)
            assert Rm / R == pytest.approx(1, abs=0.61)

    # ExpandYl

    yl = ExpandYl(H2.positions.mean(0), calc.wfs.gd, Rmax=1.5)

    def max_index(l):
        mi = 0
        limax = l[0]
        for i, li in enumerate(l):
            if limax < li:
                limax = li
                mi = i
        return mi

    # check numbers
    for n in [0, 1]:
        # gl, w = yl.expand(calc.get_pseudo_wave_function(band=n))
        gl, w = yl.expand(calc.wfs.kpt_u[0].psit_nG[n])
        parprint('max_index(gl), n=', max_index(gl), n)
        assert max_index(gl) == n

    # io
    fname = 'expandyl.dat'
    yl.to_file(calc, fname)
