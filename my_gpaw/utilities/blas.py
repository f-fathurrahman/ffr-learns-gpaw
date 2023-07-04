# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)

See also:
http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
and
http://www.netlib.org/lapack/lug/node145.html
"""
from typing import TypeVar

import numpy as np
import scipy.linalg.blas as blas

import _gpaw
from my_gpaw import debug
from my_gpaw.new import prod
from my_gpaw.typing import Array2D, ArrayND

__all__ = ['mmm']

T = TypeVar('T', float, complex)


def mmm(alpha: T,
        a: Array2D,
        opa: str,
        b: Array2D,
        opb: str,
        beta: T,
        c: Array2D) -> None:
    """Matrix-matrix multiplication using dgemm or zgemm.

    For opa='N' and opb='N', we have:::

        c <- αab + βc.

    Use 'T' to transpose matrices and 'C' to transpose and complex conjugate
    matrices.
    """

    assert opa in 'NTC'
    assert opb in 'NTC'

    if opa == 'N':
        a1, a2 = a.shape
    else:
        a2, a1 = a.shape
    if opb == 'N':
        b1, b2 = b.shape
    else:
        b2, b1 = b.shape
    assert a2 == b1
    assert c.shape == (a1, b2)

    assert a.dtype == b.dtype == c.dtype
    assert a.strides[1] == c.itemsize or a.size == 0
    assert b.strides[1] == c.itemsize or b.size == 0
    assert c.strides[1] == c.itemsize or c.size == 0
    if a.dtype == float:
        assert not isinstance(alpha, complex)
        assert not isinstance(beta, complex)
    else:
        assert a.dtype == complex

    _gpaw.mmm(alpha, a, opa, b, opb, beta, c)


def to2d(array: ArrayND) -> Array2D:
    """2D view af ndarray.

    >>> to2d(np.zeros((2, 3, 4))).shape
    (2, 12)
    """
    shape = array.shape
    return array.reshape((shape[0], prod(shape[1:])))


def mmmx(alpha: T,
         a: ArrayND,
         opa: str,
         b: ArrayND,
         opb: str,
         beta: T,
         c: ArrayND) -> None:
    """Matrix-matrix multiplication using dgemm or zgemm.

    Arrays a, b and c are converted to 2D arrays before calling mmm().
    """
    mmm(alpha, to2d(a), opa, to2d(b), opb, beta, to2d(c))


def axpy(alpha, x, y):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y

    """
    if x.size == 0:
        return
    assert x.flags.contiguous
    assert y.flags.contiguous
    x = x.ravel()
    y = y.ravel()
    if x.dtype == float:
        z = blas.daxpy(x, y, a=alpha)
    else:
        z = blas.zaxpy(x, y, a=alpha)
    assert z is y, (x, y, x.shape, y.shape)


def rk(alpha, a, beta, c, trans='c'):
    """Rank-k update of a matrix.

    For ``trans='c'`` the following operation is performed:::

              †
      c <- αaa + βc,

    and for ``trans='t'`` we get:::

             †
      c <- αa a + βc

    If the ``a`` array has more than 2 dimensions then the 2., 3., ...
    axes are combined.

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert beta == 0.0 or np.isfinite(c).all()

    assert (a.dtype == float and c.dtype == float or
            a.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous
    assert a.ndim > 1
    if trans == 'n':
        assert c.shape == (a.shape[1], a.shape[1])
    else:
        assert c.shape == (a.shape[0], a.shape[0])
    assert c.strides[1] == c.itemsize
    _gpaw.rk(alpha, a, beta, c, trans)


def r2k(alpha, a, b, beta, c, trans='c'):
    """Rank-2k update of a matrix.

    Performs the operation::

                        dag        cc       dag
      c <- alpha * a . b    + alpha  * b . a    + beta * c

    or if trans='n'::
                    dag           cc   dag
      c <- alpha * a   . b + alpha  * b   . a + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``cc`` denotes complex conjugation.

    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert beta == 0.0 or np.isfinite(np.tril(c)).all()

    assert (a.dtype == float and b.dtype == float and c.dtype == float or
            a.dtype == complex and b.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous and b.flags.contiguous
    assert a.ndim > 1
    assert a.shape == b.shape
    if trans == 'c':
        assert c.shape == (a.shape[0], a.shape[0])
    else:
        assert c.shape == (a.shape[1], a.shape[1])
    assert c.strides[1] == c.itemsize
    _gpaw.r2k(alpha, a, b, beta, c, trans)


def _gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
    """Matrix multiplication using gemm.

    return reference to out, where::

      out <- alpha * a . b + beta * out

    If out is None, a suitably sized zero array will be created.

    ``a.b`` denotes matrix multiplication, where the product-sum is
    over the last dimension of a, and either
    the first dimension of b (for trans='n'), or
    the last dimension of b (for trans='t' or 'c').

    If trans='c', the complex conjugate of b is used.
    """
    # Store original shapes
    ashape = a.shape
    bshape = b.shape

    # Vector-vector multiplication is handled by dotu
    if a.ndim == 1 and b.ndim == 1:
        assert out is None
        if trans == 'c':
            return alpha * np.vdot(b, a)  # dotc conjugates *first* argument
        else:
            return alpha * a.dot(b)

    # Map all arrays to 2D arrays
    a = a.reshape(-1, a.shape[-1])
    if trans == 'n':
        b = b.reshape(b.shape[0], -1)
        outshape = a.shape[0], b.shape[1]
    else:  # 't' or 'c'
        b = b.reshape(-1, b.shape[-1])

    # Apply BLAS gemm routine
    outshape = a.shape[0], b.shape[trans == 'n']
    if out is None:
        # (ATLAS can't handle uninitialized output array)
        out = np.zeros(outshape, a.dtype)
    else:
        out = out.reshape(outshape)
    mmmx(alpha, a, 'N', b, trans.upper(), beta, out)

    # Determine actual shape of result array
    if trans == 'n':
        outshape = ashape[:-1] + bshape[1:]
    else:  # 't' or 'c'
        outshape = ashape[:-1] + bshape[:-1]
    return out.reshape(outshape)


if not hasattr(_gpaw, 'mmm'):
    def rk(alpha, a, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'n':
            c += alpha * a.conj().T.dot(a)
        else:
            a = a.reshape((len(a), -1))
            c += alpha * a.dot(a.conj().T)

    def r2k(alpha, a, b, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'c':
            c += (alpha * a.reshape((len(a), -1))
                  .dot(b.reshape((len(b), -1)).conj().T) +
                  alpha * b.reshape((len(b), -1))
                  .dot(a.reshape((len(a), -1)).conj().T))
        else:
            c += alpha * (a.conj().T @ b + b.conj().T @ a)

    def op(o, m):
        if o == 'N':
            return m
        if o == 'T':
            return m.T
        return m.conj().T

    def mmm(alpha: T, a: np.ndarray, opa: str,  # noqa
            b: np.ndarray, opb: str,
            beta: T, c: np.ndarray) -> None:
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        c += alpha * op(opa, a).dot(op(opb, b))

    gemmdot = _gemmdot

elif not debug:
    mmm = _gpaw.mmm  # noqa
    rk = _gpaw.rk  # noqa
    r2k = _gpaw.r2k  # noqa
    gemmdot = _gemmdot

else:
    def gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert a.dtype == b.dtype
        if trans == 'n':
            assert a.shape[-1] == b.shape[0]
        else:
            assert a.shape[-1] == b.shape[-1]
        if out is not None:
            assert out.flags.contiguous
            assert a.dtype == out.dtype
            assert a.ndim > 1 or b.ndim > 1
            if trans == 'n':
                assert out.shape == a.shape[:-1] + b.shape[1:]
            else:
                assert out.shape == a.shape[:-1] + b.shape[:-1]
        return _gemmdot(a, b, alpha, beta, out, trans)
