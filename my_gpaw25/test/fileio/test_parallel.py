import pytest
from my_gpaw25.mpi import world
from my_gpaw25 import GPAW, ConvergenceError, restart
from my_gpaw25.mixer import MixerSum
from ase.build import bulk

# bulk Fe with k-point, band, and domain parallelization
pytestmark = pytest.mark.skipif(world.size < 4,
                                reason='world.size < 4')


@pytest.mark.old_gpaw_only
def test_fileio_parallel(in_tmp_dir):
    a = 2.87
    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([2.2])
    calc = GPAW(mode='fd',
                h=0.20,
                eigensolver='rmm-diis',
                mixer=MixerSum(0.1, 3),
                nbands=6,
                kpts=(4, 4, 4),
                parallel={'band': 2, 'domain': (2, 1, 1)},
                maxiter=4)
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except ConvergenceError:
        pass
    calc.write('tmp.gpw', mode='all')

    # Continue calculation for few iterations
    atoms, calc = restart('tmp.gpw',
                          eigensolver='rmm-diis',
                          mixer=MixerSum(0.1, 3),
                          parallel={'band': 2, 'domain': (1, 1, 2)},
                          maxiter=4)
    try:
        atoms.get_potential_energy()
    except ConvergenceError:
        pass
