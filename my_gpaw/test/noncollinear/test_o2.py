from unittest import SkipTest

import pytest
from ase import Atoms

from my_gpaw import GPAW
from my_gpaw.mpi import size


@pytest.mark.later
def test_noncollinear_o2(in_tmp_dir):
    if size > 2:
        raise SkipTest()

    a = Atoms('OO', [[0, 0, 0], [0, 0, 1.1]], magmoms=[1, 1], pbc=(1, 0, 0))
    a.center(vacuum=2.5)
    a.calc = GPAW(mode='pw',
                  kpts=(2, 1, 1))
    f0 = a.get_forces()

    a.calc = GPAW(mode='pw',
                  kpts=(2, 1, 1),
                  symmetry='off',
                  experimental={'magmoms': [[0, 0.5, 0.5], [0, 0, 1]]})
    f = a.get_forces()

    assert abs(f - f0).max() < 0.01

    a.calc.write('o2.gpw')
    a.calc.write('o2w.gpw', 'all')
    GPAW('o2w.gpw')
