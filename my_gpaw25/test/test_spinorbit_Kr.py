import pytest
from my_gpaw25.mpi import world
from ase import Atoms
from my_gpaw25 import GPAW
from my_gpaw25.spinorbit import soc_eigenstates

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.soc
def test_spinorbit_Kr():
    a = Atoms('Kr')
    a.center(vacuum=3.0)

    calc = GPAW(mode='pw', xc='LDA')

    a.calc = calc
    a.get_potential_energy()

    e_n = calc.get_eigenvalues()
    e_m = soc_eigenstates(calc)[0].eig_m

    assert e_n[0] - e_m[0] == pytest.approx(0.0, abs=1.0e-3)
    assert e_n[1] - e_m[2] == pytest.approx(0.452, abs=1.0e-3)
    assert e_n[2] - e_m[4] == pytest.approx(-0.226, abs=1.0e-3)
