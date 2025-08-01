from my_gpaw25.transformers import Transformer
import numpy.random as ra
from my_gpaw25.grid_descriptor import GridDescriptor


def test_fd_ops_non_periodic():
    p = 0
    n = 20
    gd1 = GridDescriptor((n, n, n), (8.0, 8.0, 8.0), pbc_c=p)
    a1 = gd1.zeros()
    rng = ra.default_rng(8)
    a1[:] = rng.random(a1.shape)
    gd2 = gd1.refine()
    a2 = gd2.zeros()
    i = Transformer(gd1, gd2).apply
    i(a1, a2)
    assert abs(gd1.integrate(a1) - gd2.integrate(a2)) < 1e-10
    r = Transformer(gd2, gd1).apply
    a2[0] = 0.0
    a2[:, 0] = 0.0
    a2[:, :, 0] = 0.0
    a2[-1] = 0.0
    a2[:, -1] = 0.0
    a2[:, :, -1] = 0.0
    r(a2, a1)
    assert abs(gd1.integrate(a1) - gd2.integrate(a2)) < 1e-10
