from ase import Atom, Atoms
from my_gpaw import GPAW
from my_gpaw.mixer import BroydenMixer
from my_gpaw.test import equal


def test_broydenmixer(in_tmp_dir):
    a = 2.7
    bulk = Atoms([Atom('Li')], pbc=True, cell=(a, a, a))
    k = 2
    g = 16
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                mixer=BroydenMixer())
    bulk.calc = calc
    e = bulk.get_potential_energy()
    calc.write('Li.gpw')
    GPAW('Li.gpw')

    energy_tolerance = 0.0001
    equal(e, -1.20258, energy_tolerance)
