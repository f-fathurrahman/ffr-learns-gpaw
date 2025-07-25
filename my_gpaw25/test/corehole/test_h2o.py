import numpy as np
import pytest

import my_gpaw25.mpi as mpi
from my_gpaw25 import GPAW
from my_gpaw25.xas import XAS


@pytest.mark.old_gpaw_only
def test_corehole_h2o(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    # Generate setup for oxygen with half a core-hole:
    calc = GPAW(gpw_files['h2o_xas'])
    if mpi.size == 1:
        xas = XAS(calc)
        x, y = xas.get_spectra()
        e1_n = xas.eps_n
        de1 = e1_n[1] - e1_n[0]

    if mpi.size == 1:
        # calc = GPAW('h2o-xas.gpw')
        # poissonsolver=FDPoissonSolver(use_charge_center=True))
        # calc.initialize()
        xas = XAS(calc)
        x, y = xas.get_spectra()
        e2_n = xas.eps_n
        w_n = np.sum(xas.sigma_cn.real**2, axis=0)
        de2 = e2_n[1] - e2_n[0]

        assert de2 == pytest.approx(2.064, abs=0.005)
        assert w_n[1] / w_n[0] == pytest.approx(2.22, abs=0.01)

        assert de1 == de2

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, y[0])
        plt.plot(x, sum(y))
        plt.show()
