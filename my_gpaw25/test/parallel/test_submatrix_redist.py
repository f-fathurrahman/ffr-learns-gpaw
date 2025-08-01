from my_gpaw25.mpi import world
from my_gpaw25.blacs import BlacsGrid, Redistributor


def test_parallel_submatrix_redist(scalapack):
    if world.size > 1:
        shape = (2, world.size // 2)
    else:
        shape = (1, 1)

    grid = BlacsGrid(world, *shape)

    desc = grid.new_descriptor(12, 8, 2, 3)

    a = desc.zeros()
    a[:] = world.rank

    subdesc = grid.new_descriptor(7, 7, 2, 2)
    b = subdesc.zeros()

    r = Redistributor(grid.comm, desc, subdesc)

    ia = 3
    ja = 2
    ib = 1
    jb = 1
    M = 4
    N = 5

    r.redistribute(a, b, M, N, ia, ja, ib, jb)

    a0 = desc.collect_on_master(a)
    b0 = subdesc.collect_on_master(b)
    if world.rank == 0:
        print(a0)
        print(b0)
        xa = a0[ia:ia + M, ja:ja + N]
        xb = b0[ib:ib + M, jb:jb + N]
        assert (xa == xb).all()
