# fails with On entry to ZGEMV parameter number 8 had an illegal value

import pytest
from my_gpaw.mpi import world
from ase.build import molecule
from my_gpaw import GPAW
from my_gpaw import PW

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_pw_moleculecg():
    m = molecule('H')
    m.center(vacuum=2.0)
    m.calc = GPAW(mode=PW(), eigensolver='cg')
    m.get_potential_energy()
