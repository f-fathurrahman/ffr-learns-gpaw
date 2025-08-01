import pytest
from my_gpaw25.mpi import world
from ase import Atoms
from my_gpaw25 import GPAW, restart
from my_gpaw25.utilities.kspot import CoreEigenvalues

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.old_gpaw_only
def test_coreeig(in_tmp_dir):
    a = 7.0
    calc = GPAW(mode='fd', h=0.1)
    system = Atoms('Ne', calculator=calc)
    system.center(vacuum=a / 2)
    e0 = system.get_potential_energy()
    calc.write('Ne.gpw')

    del calc, system

    atoms, calc = restart('Ne.gpw')
    calc.converge_wave_functions()
    e_j = CoreEigenvalues(calc).get_core_eigenvalues(0)
    assert abs(e_j[0] - (-30.344066)) * \
        27.21 < 0.1  # Error smaller than 0.1 eV

    energy_tolerance = 0.002
    assert e0 == pytest.approx(-0.0107707223, abs=energy_tolerance)
