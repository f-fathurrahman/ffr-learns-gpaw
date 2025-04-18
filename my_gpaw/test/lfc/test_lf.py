from my_gpaw.test import equal
from my_gpaw.grid_descriptor import GridDescriptor
from my_gpaw.spline import Spline
import gpaw.mpi as mpi
from my_gpaw.lfc import LocalizedFunctionsCollection as LFC


def test_lfc_lf():
    s = Spline(0, 1.0, [1.0, 0.5, 0.0])
    n = 40
    a = 8.0
    gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
    c = LFC(gd, [[s], [s], [s]])
    c.set_positions([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])
    b = gd.zeros()
    c.add(b)
    x = gd.integrate(b)

    gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
    c = LFC(gd, [[s], [s], [s]])
    c.set_positions([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])
    b = gd.zeros()
    c.add(b)
    y = gd.integrate(b)
    equal(x, y, 1e-13)
