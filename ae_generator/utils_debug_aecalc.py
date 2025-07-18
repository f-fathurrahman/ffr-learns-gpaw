from math import sqrt, pi
import numpy as np

# Needed for radial_hartree
import _gpaw 

# fine-structure constant
alpha = 1 / 137.036

# Output: u_j is modified
def initialize_wave_functions(symbol, r, dr, l_j, e_j, u_j):
    r = r
    dr = dr
    # Initialize with Slater function:
    for l, e, u in zip(l_j, e_j, u_j):
        if symbol in ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]:
            a = sqrt(-4.0 * e)
        else:
            a = sqrt(-2.0 * e)
        u[:] = r**(1 + l) * np.exp(-a * r)
        norm = np.dot(u**2, dr)
        u *= 1.0 / sqrt(norm)
    return



def calculate_density(r, f_j, u_j):
    """Return the electron charge density divided by 4 pi"""
    n = np.dot(f_j, np.where(abs(u_j) < 1e-160, 0, u_j)**2) / (4 * pi)
    n[1:] /= r[1:]**2
    n[0] = n[1]
    return n


"""Solve the Schrodinger equation

::

     2
    d u     1  dv  du   u     l(l + 1)
  - --- - ---- -- (-- - -) + [-------- + 2M(v - e)] u = 0,
      2      2 dr  dr   r         2
    dr    2Mc                    r


where the relativistic mass::

           1
  M = 1 - --- (v - e)
            2
          2c

and the fine-structure constant alpha = 1/c = 1/137.036
is set to zero for non-scalar-relativistic calculations.

On the logaritmic radial grids defined by::

      beta g
  r = ------,  g = 0, 1, ..., N - 1
      N - g

         rN
  g = --------, r = [0; oo[
      beta + r

the Schrodinger equation becomes::

   2
  d u      du
  --- c  + -- c  + u c  = 0
    2  2   dg  1      0
  dg

with the vectors c0, c1, and c2  defined by::

         2 dg 2
  c  = -r (--)
   2       dr

          2         2
         d g  2    r   dg dv
  c  = - --- r  - ---- -- --
   1       2         2 dr dr
         dr       2Mc

                            2    r   dv
  c  = l(l + 1) + 2M(v - e)r  + ---- --
   0                               2 dr
                                2Mc
"""
# XXX Modified?
def solve_radial_schrod(rgd, N, r, dr, vr, d2gdr2, n_j, l_j,e_j, u_j, scalarrel=True):

    c2 = -(r / dr)**2
    c10 = -d2gdr2 * r**2  # first part of c1 vector

    if scalarrel:
        r2dvdr = np.zeros(N)
        rgd.derivative(vr, r2dvdr)
        r2dvdr *= r
        r2dvdr -= vr
    else:
        r2dvdr = None
    # XXX: r2dvdr is local to this function

    # solve for each quantum state separately
    for j, (n, l, e, u) in enumerate(zip(n_j, l_j, e_j, u_j)):
        nodes = n - l - 1  # analytically expected number of nodes
        delta = -0.2 * e
        nn, A = eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)
        # adjust eigenenergy until u has the correct number of nodes
        while nn != nodes:
            diff = np.sign(nn - nodes)
            while diff == np.sign(nn - nodes):
                e -= diff * delta
                nn, A = eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)
            delta /= 2

        # adjust eigenenergy until u is smooth at the turning point
        de = 1.0
        while abs(de) > 1e-9:
            norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
            u *= 1.0 / sqrt(norm)
            de = 0.5 * A / norm
            x = abs(de / e)
            if x > 0.1:
                de *= 0.1 / x
            e -= de
            assert e < 0.0
            nn, A = eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)
        e_j[j] = e
        u *= 1.0 / sqrt(np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr))

    return



"""
n, A = shoot(u, l, vr, e, ...)

For guessed trial eigenenergy e, integrate the radial Schrodinger
equation::

  2
 d u      du
 --- c  + -- c  + u c  = 0
   2  2   dg  1      0
 dg

       2 dg 2
c  = -r (--)
 2       dr

                2         2
       d g  2    r   dg dv
c  = - --- r  - ---- -- --
 1       2         2 dr dr
       dr       2Mc

                          2    r   dv
c  = l(l + 1) + 2M(v - e)r  + ---- --
 0                               2 dr
                              2Mc

The resulting wavefunction is returned in input vector u.
The number of nodes of u is returned in attribute n.
Returned attribute A, is a measure of the size of the derivative
discontinuity at the classical turning point.
The trial energy e is correct if A is zero and n is the correct number
of nodes."""

def eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel, gmax=None):

    if scalarrel:
        x = 0.5 * alpha**2  # x = 1 / (2c^2)
        Mr = r * (1.0 + x * e) - x * vr
    else:
        Mr = r
    c0 = l * (l + 1) + 2 * Mr * (vr - e * r)
    if gmax is None and np.all(c0 > 0):
        raise ConvergenceError("Bad initial electron density guess!")

    c1 = c10
    if scalarrel:
        c0 += x * r2dvdr / Mr
        c1 = c10 - x * r * r2dvdr / (Mr * dr)

    # vectors needed for numeric integration of diff. equation
    fm = 0.5 * c1 - c2
    fp = 0.5 * c1 + c2
    f0 = c0 - 2 * c2

    if gmax is None:
        # set boundary conditions at r -> oo (u(oo) = 0 is implicit)
        u[-1] = 1.0

        # perform backwards integration from infinity to the turning point
        g = len(u) - 2
        u[-2] = u[-1] * f0[-1] / fm[-1]
        while c0[g] > 0.0:  # this defines the classical turning point
            u[g - 1] = (f0[g] * u[g] + fp[g] * u[g + 1]) / fm[g]
            if u[g - 1] < 0.0:
                # There should"t be a node here!  Use a more negative
                # eigenvalue:
                print("!!!!!!", end=" ")
                return 100, None
            if u[g - 1] > 1e100:
                u *= 1e-100
            g -= 1

        # stored values of the wavefunction and the first derivative
        # at the turning point
        gtp = g + 1
        utp = u[gtp]
        if gtp == len(u) - 1:
            return 100, 0.0
        dudrplus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    else:
        gtp = gmax

    # set boundary conditions at r -> 0
    u[0] = 0.0
    u[1] = 1.0

    # perform forward integration from zero to the turning point
    g = 1
    nodes = 0
    # integrate one step further than gtp
    # (such that dudr is defined in gtp)
    while g <= gtp:
        u[g + 1] = (fm[g] * u[g - 1] - f0[g] * u[g]) / fp[g]
        if u[g + 1] * u[g] < 0:
            nodes += 1
        g += 1
    if gmax is not None:
        return

    # scale first part of wavefunction, such that it is continuous at gtp
    u[:gtp + 2] *= utp / u[gtp]

    # determine size of the derivative discontinuity at gtp
    dudrminus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    A = (dudrplus - dudrminus) * utp

    return nodes, A


# Radial-grid Hartree solver:
#
#                       l
#             __  __   r
#     1      \   4||    <   * ^    ^
#   ------ =  )  ---- ---- Y (r)Y (r"),
#    _ _     /__ 2l+1  l+1  lm   lm
#   |r-r"|    lm      r
#                      >
# where
#
#   r = min(r, r")
#    <
#
# and
#
#   r = max(r, r")
#    >
#
"""
Calculates radial Coulomb integral.

The following integral is calculated::

                               ^
                      n (r")Y (r")
          ^    / _     l     lm
  v (r)Y (r) = |dr" --------------,
   l    lm     /        _   _
                       |r - r"|

where input and output arrays `nrdr` and `vr`::

          dr
  n (r) r --  and  v (r) r.
   l      dg        l
"""
def radial_hartree(l: int, nrdr: np.ndarray, r: np.ndarray, vr: np.ndarray) -> None:

    assert is_contiguous(nrdr, float)
    assert is_contiguous(r, float)
    assert is_contiguous(vr, float)
    assert nrdr.shape == vr.shape and len(vr.shape) == 1
    assert len(r.shape) == 1
    assert len(r) >= len(vr)
    return _gpaw.hartree(l, nrdr, r, vr)


def is_contiguous(array, dtype=None):
    """Check for contiguity and type."""
    if dtype is None:
        return array.flags.c_contiguous
    else:
        return array.flags.c_contiguous and array.dtype == dtype


def py_radial_hartree(l: int, nrdr: np.ndarray, r: np.ndarray, vr: np.ndarray) -> None:

    M = nrdr.shape[0]

    p = 0.0
    q = 0.0
    for g in range(M-1, 0, -1):
        R = r[g]
        rl = R**l
        dp = nrdr[g]/rl
        rlp1 = rl * R
        dq = nrdr[g] * rlp1
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl
        p += dp
        q += dq

    #print(f"p = {p} q = {q}")
    
    vr[0] = 0.0
    f = 4.0*pi / (2*l + 1)
    #for (int g = 1; g < M; g++)
    for g in range(1,M):
        R = r[g]
        vr[g] = f * (vr[g] + q / R**l)

    return


def radial_kinetic_energy_density(rgd, r, f_j, l_j, u_j):
    """Kinetic energy density from a restricted set of wf's
    """
    shape = np.shape(u_j[0])
    dudr = np.zeros(shape)
    tau = np.zeros(shape)
    for f, l, u in zip(f_j, l_j, u_j):
        rgd.derivative(u, dudr)
        # contribution from angular derivatives
        if l > 0:
            tau += f * l * (l + 1) * np.where(abs(u) < 1e-160, 0, u)**2
        # contribution from radial derivatives
        dudr = u - r * dudr
        tau += f * np.where(abs(dudr) < 1e-160, 0, dudr)**2
    tau[1:] /= r[1:]**4
    tau[0] = tau[1]

    return 0.5 * tau / (4 * pi)