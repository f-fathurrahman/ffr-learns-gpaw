# fails with On entry to ZGEMV parameter number 8 had an illegal value

import pytest
from my_gpaw25.mpi import world
from ase.build import molecule
from my_gpaw25 import GPAW
from my_gpaw25 import PW

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_pw_moleculecg():
    m = molecule('H')
    m.center(vacuum=2.0)
    m.calc = GPAW(mode=PW(), eigensolver='cg')
    m.get_potential_energy()
