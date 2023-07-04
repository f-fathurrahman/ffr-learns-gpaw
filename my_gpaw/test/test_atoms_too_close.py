import pytest
from ase import Atoms
from my_gpaw import GPAW
from my_gpaw.utilities import AtomsTooClose


def test_atoms_too_close():
    atoms = Atoms('H2', [(0.0, 0.0, 0.0),
                         (0.0, 0.0, 3.995)],
                  cell=(4, 4, 4), pbc=True)

    calc = GPAW(txt=None)
    atoms.calc = calc

    with pytest.raises(AtomsTooClose):
        atoms.get_potential_energy()
