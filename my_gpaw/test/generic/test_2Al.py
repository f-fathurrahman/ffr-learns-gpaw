from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal


def test_generic_2Al():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0, 0, d))],
                 cell=(4 * d, 4 * d, 2 * d),
                 pbc=1)
    n = 16
    calc = GPAW(gpts=(2 * n, 2 * n, 1 * n),
                nbands=1 * 8,
                kpts=(1, 1, 4),
                convergence={'eigenstates': 2.3e-9})
    bulk.calc = calc
    e2 = bulk.get_potential_energy()

    bulk = bulk.repeat((1, 1, 2))
    calc = GPAW(gpts=(2 * n, 2 * n, 2 * n),
                nbands=16,
                kpts=(1, 1, 2),
                convergence={'eigenstates': 2.3e-9})
    bulk.calc = calc
    e4 = bulk.get_potential_energy()

    # checks
    energy_tolerance = 0.002

    print(e2, e4)
    equal(e4 / 2, e2, 48e-6)
    equal(e2, -3.41595, energy_tolerance)
    equal(e4, -6.83191, energy_tolerance)
