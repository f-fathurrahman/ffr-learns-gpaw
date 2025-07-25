def syrk(trans, a, c, alpha, beta, lower):
    from my_gpaw25.utilities.blas import rk

    assert trans == 'N'
    assert lower
    rk(alpha, a._data, beta, c._data)


def gemm(transa, transb, a, b, c, alpha, beta):
    from my_gpaw25.utilities.blas import mmm

    mmm(alpha,
        a._data,
        transa.replace('H', 'C'),
        b._data,
        transb.replace('H', 'C'),
        beta,
        c._data)
