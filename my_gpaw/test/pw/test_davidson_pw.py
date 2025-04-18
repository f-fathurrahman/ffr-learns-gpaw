import pytest
from my_gpaw.mpi import world
from ase import Atom, Atoms
from my_gpaw import GPAW
from my_gpaw.test import equal

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_pw_davidson_pw():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.5, 0.5, 0.5))], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    calc = GPAW(mode='pw',
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'eigenstates': 7.2e-9, 'energy': 1e-5})
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    niter0 = calc.get_number_of_iterations()
    calc = GPAW(mode='pw',
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'eigenstates': 7.2e-9,
                             'energy': 1e-5,
                             'bands': 5},
                eigensolver='dav')
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    niter1 = calc.get_number_of_iterations()
    equal(e0, e1, 5.0e-6)

    energy_tolerance = 0.0004
    equal(e0, -6.97798, energy_tolerance)
    assert 8 <= niter0 <= 12, niter0
    equal(e1, -6.97798, energy_tolerance)
    assert 8 <= niter1 <= 22, niter1
