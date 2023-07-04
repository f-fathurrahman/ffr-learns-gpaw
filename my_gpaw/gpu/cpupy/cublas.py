from gpaw.utilities.blas import rk
from gpaw.utilities.blas import mmm


def syrk(trans, a, c, alpha, beta, lower):
    assert trans == 'N'
    assert lower
    rk(alpha, a._data, beta, c._data)


def gemm(transa, transb, a, b, c, alpha, beta):
    mmm(alpha,
        a._data,
        transa.replace('H', 'C'),
        b._data,
        transb.replace('H', 'C'),
        beta,
        c._data)
