from my_gpaw.fd_operators import Gradient, FDOperator
import numpy as np
from my_gpaw.grid_descriptor import GridDescriptor
from my_gpaw.mpi import world


def test_fd_ops_gradient():
    if world.size > 4:
        # Grid is so small that domain decomposition cannot exceed 4 domains
        assert world.size % 4 == 0
        group, other = divmod(world.rank, 4)
        ranks = np.arange(4 * group, 4 * (group + 1))
        domain_comm = world.new_communicator(ranks)
    else:
        domain_comm = world

    gd = GridDescriptor((8, 1, 1), (8.0, 1.0, 1.0), comm=domain_comm)
    a = gd.zeros()
    dadx = gd.zeros()
    a[:, 0, 0] = np.arange(gd.beg_c[0], gd.end_c[0])
    gradx = Gradient(gd, v=0)
    print(a.itemsize, a.dtype, a.shape)
    print(dadx.itemsize, dadx.dtype, dadx.shape)
    gradx.apply(a, dadx)

    #   a = [ 0.  1.  2.  3.  4.  5.  6.  7.]

    dAdx = gd.collect(dadx, broadcast=True)
    assert not (dAdx[:, 0, 0] - [-3, 1, 1, 1, 1, 1, 1, -3]).any()

    # Backwards FD operator:
    gradx2 = FDOperator([-1, 1], [[-1, 0, 0], [0, 0, 0]], gd)
    gradx2.apply(a, dadx)
    dAdx = gd.collect(dadx, broadcast=True)
    assert not (dAdx[:, 0, 0] - [-7, 1, 1, 1, 1, 1, 1, 1]).any()

    gd = GridDescriptor((1, 8, 1), (1.0, 8.0, 1.0),
                        (1, 0, 1), comm=domain_comm)
    dady = gd.zeros()
    a = gd.zeros()
    grady = Gradient(gd, v=1)
    a[0, :, 0] = np.arange(gd.beg_c[1], gd.end_c[1]) - 1
    grady.apply(a, dady)

    #   da
    #   -- = [0.5  1.   1.   1.   1.   1.  -2.5]
    #   dy

    dady = gd.collect(dady, broadcast=True)
    assert dady[0, 0, 0] == 0.5 and np.sum(dady[0, :, 0]) == 3.0

    # a GUC cell
    gd = GridDescriptor((1, 7, 1),
                        ((1.0, 0.0, 0.0),
                         (5.0, 5.0, 0.0),
                         (0.0, 0.0, 0.7)), comm=domain_comm)
    dady = gd.zeros()
    grady = Gradient(gd, v=1)
    a = gd.zeros()
    a[0, :, 0] = np.arange(gd.beg_c[1], gd.end_c[1]) - 1
    grady.apply(a, dady)

    #   da
    #   -- = [-3.5  1.4   1.4   1.4   1.4   1.4  -3.5]
    #   dy

    dady = gd.collect(dady, broadcast=True)
    assert abs(dady[0, 0, 0] - -
               3.5) < 1e-12 and abs(np.sum(dady[0, :, 0])) < 1e-12

    # Check continuity of weights:
    weights = []
    for x in np.linspace(-0.6, 0.6, 130, True):
        gd = GridDescriptor((3, 3, 3),
                            ((3.0, 0.0, 0.0),
                             (3 * x, 1.5 * 3**0.5, 0.0),
                             (0.0, 0.0, 3)),
                            comm=domain_comm)
        g = Gradient(gd, v=0)
        for c, o in zip(g.coef_p, g.offset_pc):
            if (o == (-1, 1, 0)).all():
                break
        else:
            c = 0.0
        weights.append(c)

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(weights)
        plt.show()
