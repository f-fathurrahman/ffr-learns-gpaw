from ase import Atom, Atoms
from my_gpaw import GPAW
from my_gpaw.eigensolvers.davidson import Davidson
from my_gpaw.mpi import size
from my_gpaw.test import equal


def test_eigen_davidson():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.5, 0.5, 0.5))], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    h = 0.25
    calc = GPAW(h=h,
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'eigenstates': 7.2e-9, 'energy': 1e-5})
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    calc = GPAW(h=h,
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'eigenstates': 7.2e-9,
                             'energy': 1e-5,
                             'bands': 5},
                eigensolver='dav')
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    equal(e0, e1, 5.0e-5)

    energy_tolerance = 0.0004
    equal(e0, -6.97626, energy_tolerance)
    equal(e1, -6.976265, energy_tolerance)

    # band parallelization
    if size % 2 == 0:
        calc = GPAW(h=h,
                    nbands=2 * 8,
                    kpts=(2, 2, 2),
                    convergence={'eigenstates': 7.2e-9,
                                 'energy': 1e-5,
                                 'bands': 5},
                    parallel={'band': 2},
                    eigensolver=Davidson(niter=3))
        bulk.calc = calc
        e3 = bulk.get_potential_energy()
        equal(e0, e3, 5.0e-5)
