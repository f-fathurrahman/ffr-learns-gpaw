"""Test BLAS matrix-matrix-multiplication interface."""
import numpy as np
from my_gpaw25.utilities.blas import mmm


def test_linalg_mmm(rng):
    def op(o, m):
        if o == 'N':
            return m
        if o == 'T':
            return m.T
        return m.T.conj()

    def matrix(shape, dtype):
        if dtype == float:
            return rng.random(shape)
        return rng.random(shape) + 1j * rng.random(shape)

    for dtype in [float, complex]:
        a = matrix((2, 3), dtype)
        for opa in 'NTC':
            A = op(opa, a)
            B = matrix((A.shape[1], 4), dtype)
            for opb in 'NTC':
                b = op(opb, B).copy()
                C = np.dot(A, B)
                mmm(1, a, opa, b, opb, -1, C)
                assert abs(C).max() < 1e-14
