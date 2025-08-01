# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from my_gpaw25.utilities.blas import axpy
from my_gpaw25.mpi import rank

from .base import BaseSolver


class CSCG(BaseSolver):
    """Conjugate gradient for complex symmetric matrices

    This class solves a set of linear equations A.x = b using conjugate
    gradient for complex symmetric matrices. The matrix A is a complex,
    symmetric, and non-singular matrix. The method requires only access
    to matrix-vector product A.x = b, which is called A.dot(x). Thus A
    must provide the member function dot(self,x,b), where x and b are
    complex arrays (numpy.array([], complex), and x is the known vector,
    and b is the result.

    Now x and b are multivectors, i.e., list of vectors.
    """

    def solve(self, A, x, b):
        if self.timer is not None:
            self.timer.start('CSCG')

        # number of vectors
        nvec = len(x)

        # r_0 = b - A x_0
        r = self.gd.zeros(nvec, dtype=complex)
        A.dot(-x, r)
        r += b

        p = self.gd.zeros(nvec, dtype=complex)
        q = self.gd.zeros(nvec, dtype=complex)
        z = self.gd.zeros(nvec, dtype=complex)

        alpha = np.zeros((nvec,), dtype=complex)
        beta = np.zeros((nvec,), dtype=complex)
        rho = np.zeros((nvec,), dtype=complex)
        rhop = np.zeros((nvec,), dtype=complex)
        scale = np.zeros((nvec,), dtype=complex)
        tmp = np.zeros((nvec,), dtype=complex)

        rhop[:] = 1.

        # Multivector dot product, a^T b, where ^T is transpose
        def multi_zdotu(s, x, y, nvec):
            for i in range(nvec):
                s[i] = x[i].ravel().dot(y[i].ravel())
                # s[i] = dotu(x[i],y[i])
            self.gd.comm.sum(s)
            return s

        # Multivector ZAXPY: a x + y => y
        def multi_zaxpy(a, x, y, nvec):
            for i in range(nvec):
                axpy(a[i] * (1 + 0J), x[i], y[i])

        # Multiscale: a x => x
        def multi_scale(a, x, nvec):
            for i in range(nvec):
                x[i] *= a[i]

        # scale = square of the norm of b
        multi_zdotu(scale, b, b, nvec)
        scale = np.abs(scale)

        # if scale < eps, then convergence check breaks down
        if (scale < self.eps).any():
            raise RuntimeError(
                "CSCG method detected underflow for squared norm of "
                "right-hand side (scale = %le < eps = %le)." %
                (scale, self.eps))

        # print 'Scale = ', scale

        slow_convergence_iters = 100

        for i in range(self.max_iter):
            # z_i = (M^-1.r)
            A.apply_preconditioner(r, z)

            # rho_i-1 = r^T z_i-1
            multi_zdotu(rho, r, z, nvec)

            # print 'Rho = ', rho

            # if i=1, p_i = r_i-1
            # else beta = (rho_i-1 / rho_i-2) (alpha_i-1 / omega_i-1)
            #      p_i = r_i-1 + b_i-1 (p_i-1 - omega_i-1 v_i-1)
            beta = rho / rhop

            # print 'Beta = ', beta

            # if abs(beta) / scale < eps, then CSCG breaks down
            if ((i > 0) and
                ((np.abs(beta) / scale) < self.eps).any()):
                raise RuntimeError(
                    "Conjugate gradient method failed "
                    "(abs(beta)=%le < eps = %le)." %
                    (np.min(np.abs(beta)), self.eps))

            # p = z + beta p
            multi_scale(beta, p, nvec)
            p += z

            # q = A.p
            A.dot(p, q)

            # alpha_i = rho_i-1 / (p^T q_i)
            multi_zdotu(alpha, p, q, nvec)
            alpha = rho / alpha

            # print 'Alpha = ', alpha

            # x_i = x_i-1 + alpha_i p_i
            multi_zaxpy(alpha, p, x, nvec)
            # r_i = r_i-1 - alpha_i q_i
            multi_zaxpy(-alpha, q, r, nvec)

            # if ( |r|^2 < tol^2 ) done
            multi_zdotu(tmp, r, r, nvec)
            if ((np.abs(tmp) / scale) < self.tol * self.tol).all():
                # print 'R2 of proc #', rank, '  = ' , tmp, \
                #     ' after ', i+1, ' iterations'
                break

            # print if slow convergence
            if ((i + 1) % slow_convergence_iters) == 0:
                print('R2 of proc #', rank, '  = ', tmp,
                      ' after ', i + 1, ' iterations')

            # finally update rho
            rhop[:] = rho

        # if max iters reached, raise error
        if (i >= self.max_iter - 1):
            raise RuntimeError(
                "Conjugate gradient method failed to converged "
                "within given number of iterations (= %d)." % self.max_iter)

        # done
        self.iterations = i + 1
        # print 'CSCG iterations = ', self.iterations

        if self.timer is not None:
            self.timer.stop('CSCG')

        return self.iterations
        # print self.iterations
