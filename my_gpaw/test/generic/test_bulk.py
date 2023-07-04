from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal
import numpy as np


def test_generic_bulk():
    bulk = Atoms([Atom('Li')], pbc=True)
    k = 4
    g = 8
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2)
    bulk.calc = calc
    a = np.linspace(2.6, 2.8, 5)
    e = []
    for x in a:
        bulk.set_cell((x, x, x))
        e1 = bulk.get_potential_energy()
        e.append(e1)

    fit = np.polyfit(a, e, 2)
    a0 = np.roots(np.polyder(fit, 1))[0]
    e0 = np.polyval(fit, a0)
    print('a,e =', a0, e0)
    equal(a0, 2.641, 0.001)
    equal(e0, -1.98357, 0.0002)

    energy_tolerance = 0.0002
    equal(e1, -1.96157, energy_tolerance)
