# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import pytest
import numpy as np
from my_gpaw25.grid_descriptor import GridDescriptor
from my_gpaw25.transformers import Transformer


@pytest.mark.ci
def test_fd_ops_transformations():
    n = 20
    gd = GridDescriptor((n, n, n))
    rng = np.random.RandomState(8)
    a = gd.empty()
    a[:] = rng.random(a.shape)

    gd2 = gd.refine()
    b = gd2.zeros()
    for k in [2, 4, 6, 8]:
        inter = Transformer(gd, gd2, k // 2).apply
        inter(a, b)
        assert abs(gd.integrate(a) - gd2.integrate(b)) < 1e-14

    gd2 = gd.coarsen()
    b = gd2.zeros()
    for k in [2, 4, 6, 8]:
        restr = Transformer(gd, gd2, k // 2).apply
        restr(a, b)
        assert abs(gd.integrate(a) - gd2.integrate(b)) < 1e-14

    # complex versions
    a = gd.empty(dtype=complex)
    a.real = rng.random(a.shape)
    a.imag = rng.random(a.shape)

    phase = np.ones((3, 2), complex)

    gd2 = gd.refine()
    b = gd2.zeros(dtype=complex)
    for k in [2, 4, 6, 8]:
        inter = Transformer(gd, gd2, k // 2, complex).apply
        inter(a, b, phase)
        assert abs(gd.integrate(a) - gd2.integrate(b)) < 1e-14

    gd2 = gd.coarsen()
    b = gd2.zeros(dtype=complex)
    for k in [2, 4, 6, 8]:
        restr = Transformer(gd, gd2, k // 2, complex).apply
        restr(a, b, phase)
        assert abs(gd.integrate(a) - gd2.integrate(b)) < 1e-14
