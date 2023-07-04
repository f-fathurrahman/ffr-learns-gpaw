from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal


def test_eigen_cg():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.5, 0.5, 0.5))],
                 pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    h = 0.25
    calc = GPAW(h=h,
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'energy': 1e-5})
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    calc = GPAW(h=h,
                nbands=2 * 8,
                kpts=(2, 2, 2),
                convergence={'energy': 1e-5},
                eigensolver='cg')
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    equal(e0, e1, 5.e-5)

    energy_tolerance = 0.001
    equal(e0, -6.97626, energy_tolerance)
    equal(e1, -6.97627, energy_tolerance)
