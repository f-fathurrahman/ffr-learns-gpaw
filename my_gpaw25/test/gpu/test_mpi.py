import pytest

import my_gpaw25.cgpaw as cgpaw
from my_gpaw25.gpu import cupy as cp
from my_gpaw25.gpu.mpi import CuPyMPI
import my_gpaw25.mpi as mpi


@pytest.mark.gpu
def test_mpi():
    a = cp.ones(1)
    world = mpi.world
    print(a, world, a.shape, a.nbytes, a.size, a.dtype, a.data.ptr)
    if not getattr(cgpaw, 'gpu_aware_mpi', False):
        world = CuPyMPI(world)
    world.sum(a)
    assert a[0].get() == world.size
