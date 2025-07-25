from math import pi, sqrt
import numpy as np
from my_gpaw25.utilities.tools import coordinates
from my_gpaw25.utilities.gauss import Gaussian
from my_gpaw25.grid_descriptor import GridDescriptor
import pytest
from my_gpaw25.poisson import PoissonSolver


def test_gauss_func():
    def norm(a):
        return np.sqrt(np.sum(a.ravel()**2)) / len(a.ravel())

    # Initialize classes
    a = 20  # Size of cell
    N = 48  # Number of grid points
    Nc = (N, N, N)  # Number of grid points along each axis
    gd = GridDescriptor(Nc, (a, a, a), 0)  # Grid-descriptor object
    solver = PoissonSolver(nn=3)  # Numerical poisson solver
    solver.set_grid_descriptor(gd)
    solve = solver.solve
    xyz, r2 = coordinates(gd)  # Matrix with square of the radial coordinate
    print(r2.shape)
    r = np.sqrt(r2)  # Matrix with the values of the radial coordinate
    nH = np.exp(-2 * r) / pi     # Density of the hydrogen atom
    gauss = Gaussian(gd)          # An instance of Gaussian

    # /------------------------------------------------\
    # | Check if Gaussian densities are made correctly |
    # \------------------------------------------------/
    for gL in range(2, 9):
        g = gauss.get_gauss(gL)  # a gaussian of gL'th order
        print('\nGaussian of order', gL)
        for mL in range(9):
            m = gauss.get_moment(g, mL)  # the mL'th moment of g
            print(f'  {mL}\'th moment = {m:2.6f}')
            assert m == pytest.approx(gL == mL, abs=1e-4)

    # Check the moments of the constructed 1s density
    print('\nDensity of Hydrogen atom')
    for L in range(4):
        m = gauss.get_moment(nH, L)
        print(f'  {L}\'th moment = {m:2.6f}')
        assert m == pytest.approx((L == 0) / sqrt(4 * pi), abs=1.5e-3)

    # Check that it is removed correctly
    gauss.remove_moment(nH, 0)
    m = gauss.get_moment(nH, 0)
    print('\nZero\'th moment of compensated Hydrogen density =', m)
    assert m == pytest.approx(0., abs=1e-7)

    # /-------------------------------------------------\
    # | Check if Gaussian potentials are made correctly |
    # \-------------------------------------------------/

    # Array for storing the potential
    pot = gd.zeros(dtype=float, global_array=False)
    for L in range(7):  # Angular index of gaussian
        # Get analytic functions
        ng = gauss.get_gauss(L)
        vg = gauss.get_gauss_pot(L)

        # Solve potential numerically
        solve(pot, ng, charge=None, zero_initial_phi=True)

        # Determine residual
        residual = norm(pot - vg)
        residual = gd.integrate((pot - vg)**2)**0.5

        # print result
        print('L={}, processor {} of {}: {}'.format(
            L,
            gd.comm.rank + 1,
            gd.comm.size,
            residual))

        assert residual < 0.6

    # mpirun -np 2 python gauss_func.py --gpaw-parallel --gpaw-debug
