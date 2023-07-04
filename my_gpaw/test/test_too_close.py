"""Make sure we get an exception when an atom is too close to the boundary."""
import os
from ase import Atoms
from my_gpaw import GPAW
from my_gpaw.grid_descriptor import GridBoundsError
from my_gpaw.utilities import AtomsTooClose
import pytest


@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_too_close_to_boundary(mode):
    if mode == 'pw' and not os.environ.get('GPAW_NEW'):
        return
    a = 4.0
    x = 0.1
    hydrogen = Atoms('H', [(x, x, x)],
                     cell=(a, a, a),
                     pbc=(1, 1, 0))
    hydrogen.calc = GPAW(mode=mode)
    with pytest.raises((GridBoundsError, AtomsTooClose)):
        hydrogen.get_potential_energy()
