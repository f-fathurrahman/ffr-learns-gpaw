import numpy as np
from math import pi
from my_gpaw25.coulomb import Coulomb
from my_gpaw25.grid_descriptor import GridDescriptor
from my_gpaw25.mpi import world
from my_gpaw25.utilities.gauss import coordinates
import pytest
import time


def test_coulomb():
    def test_coulomb(N=2**6, a=20):
        Nc = (N, N, N)            # Number of grid point
        gd = GridDescriptor(Nc, (a, a, a), True)  # grid-descriptor object
        # matrix with the square of the radial coordinate
        xyz, r2 = coordinates(gd)
        # matrix with the values of the radial coordinate
        r = np.sqrt(r2)
        nH = np.exp(-2 * r) / pi  # density of the hydrogen atom
        C = Coulomb(gd)           # coulomb calculator

        if world.size > 1:
            C.load('real')
            t0 = time.time()
            print('Processor {} of {}: {} Ha in {} sec'.format(
                gd.comm.rank + 1,
                gd.comm.size,
                -0.5 * C.coulomb(nH, method='real'),
                time.time() - t0))
            return
        else:
            C.load('recip_ewald')
            C.load('recip_gauss')
            C.load('real')
            test = {}
            t0 = time.time()
            test['dual density'] = (-0.5 * C.coulomb(nH, nH.copy()),
                                    time.time() - t0)
            for method in ('real', 'recip_gauss', 'recip_ewald'):
                t0 = time.time()
                test[method] = (-0.5 * C.coulomb(nH, method=method),
                                time.time() - t0)
            return test

    analytic = -5 / 16.0
    res = test_coulomb(N=48, a=15)
    if world.size == 1:
        print('Units: Bohr and Hartree')
        print('%12s %8s %8s' % ('Method', 'Energy', 'Time'))
        print('%12s %2.6f %6s' % ('analytic', analytic, '--'))
        for method, et in res.items():
            print('%12s %2.6f %1.7f' % ((method,) + et))

        assert res['real'][0] == pytest.approx(analytic, abs=6e-3)
        assert res['recip_gauss'][0] == pytest.approx(analytic, abs=6e-3)
        assert res['recip_ewald'][0] == pytest.approx(analytic, abs=2e-2)
        assert res['dual density'][0] == pytest.approx(res['recip_gauss'][0],
                                                       abs=1e-9)

    # mpirun -np 2 python coulomb.py --gpaw-parallel --gpaw-debug
