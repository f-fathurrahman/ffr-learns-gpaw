
from my_gpaw25.utilities.tools import coordinates
from my_gpaw25.utilities.gauss import Gaussian
from my_gpaw25.grid_descriptor import GridDescriptor
from my_gpaw25.helmholtz import HelmholtzSolver, ScreenedPoissonGaussian

# Initialize classes


def test_poisson_screened_poisson():
    a = 20  # Size of cell
    inv_width = 31  # inverse width of the gaussian
    N = 48  # Number of grid points
    coupling = -0.4  # dampening
    Nc = (N, N, N)                # Number of grid points along each axis
    gd = GridDescriptor(Nc, (a, a, a), 0)    # Grid-descriptor object
    solver = HelmholtzSolver(k2=coupling, nn=3)  # Numerical poisson solver
    # solver = PoissonSolver(nn=3)  # Numerical poisson solver
    # solver = HelmholtzSolver(0.16)  # Numerical poisson solver
    solver.set_grid_descriptor(gd)
    # Matrix with the square of the radial coordinate
    xyz, r2 = coordinates(gd)
    gauss = Gaussian(gd, a=inv_width)          # An instance of Gaussian
    test_screened_poisson = ScreenedPoissonGaussian(gd, a=inv_width)

    # Check if Gaussian potentials are made correctly

    # Array for storing the potential
    pot = gd.zeros(dtype=float, global_array=False)
    solver.load_gauss()
    vg = test_screened_poisson.get_phi(-coupling)  # esp. for dampening
    # Get analytic functions
    ng = gauss.get_gauss(0)
    #    vg = solver.phi_gauss
    # Solve potential numerically
    solver.solve(pot, ng, charge=None, zero_initial_phi=True)
    # Determine residual
    # residual = norm(pot - vg)
    residual = gd.integrate((pot - vg)**2)**0.5

    # print result
    print('residual %s' % (
        residual))
    assert residual < 0.003

    # mpirun -np 2 python gauss_func.py --gpaw-parallel --gpaw-debug
