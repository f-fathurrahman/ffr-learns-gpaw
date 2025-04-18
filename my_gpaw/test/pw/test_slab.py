import pytest
from ase import Atoms
from ase.optimize import BFGS

from my_gpaw import PW, SCIPY_VERSION
from my_gpaw.new.ase_interface import GPAW as NewGPAW
from my_gpaw.calculator import GPAW as OldGPAW
from my_gpaw.mpi import world


@pytest.mark.parametrize(
    'gpu, GPAW',
    [(False, OldGPAW),
     (False, NewGPAW),
     pytest.param(True, NewGPAW, marks=pytest.mark.gpu)])
def test_pw_slab(gpu, GPAW):
    if GPAW is NewGPAW and world.size > 1:
        pytest.skip()
    if gpu and SCIPY_VERSION < [1, 6]:
        pytest.skip()

    a = 2.65
    slab = Atoms('Li2',
                 [(0, 0, 0), (0, 0, a)],
                 cell=(a, a, 3 * a),
                 pbc=True)
    k = 4
    parallel = {'band': min(world.size, 4)}
    if gpu:
        parallel['gpu'] = True
    calc = GPAW(mode=PW(200),
                eigensolver='rmm-diis',
                parallel=parallel,
                kpts=(k, k, 1))
    slab.calc = calc
    BFGS(slab).run(fmax=0.01)
    assert abs(slab.get_distance(0, 1) - 2.46) < 0.01
