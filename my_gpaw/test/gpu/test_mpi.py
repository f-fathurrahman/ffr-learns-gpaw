import pytest

import _gpaw
from my_gpaw.gpu import cupy as cp
from my_gpaw.gpu.mpi import CuPyMPI
import gpaw.mpi as mpi


@pytest.mark.gpu
def test_mpi():
    a = cp.ones(1)
    world = mpi.world
    print(a, world, a.shape, a.nbytes, a.size, a.dtype, a.data.ptr)
    if not getattr(_gpaw, 'gpu_aware_mpi', False):
        world = CuPyMPI(world)
    world.sum(a)
    assert a[0].get() == world.size
