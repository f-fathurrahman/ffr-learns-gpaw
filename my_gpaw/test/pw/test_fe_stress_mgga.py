import numpy as np
import pytest
from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW, FermiDirac


@pytest.mark.mgga
def test_pw_fe_stress_mgga(in_tmp_dir):
    xc = 'revTPSS'
    m = [2.9]
    fe = bulk('Fe')
    fe.set_initial_magnetic_moments(m)
    k = 3
    fe.calc = GPAW(mode=PW(800),
                   h=0.15,
                   occupations=FermiDirac(width=0.03),
                   xc=xc,
                   kpts=(k, k, k),
                   convergence={'energy': 1e-8},
                   parallel={'domain': 1, 'augment_grids': True},
                   txt='fe.txt')

    fe.set_cell(np.dot(fe.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    fe.get_potential_energy()

    # Trigger nasty bug (fixed in !486):
    fe.calc.wfs.pt.blocksize = fe.calc.wfs.pd.maxmyng - 1

    s_analytical = fe.get_stress()
    # Calculated numerical stress once, store here to speed up test
    # numerical stresses:
    # revTPSS stress: [0.03113369 -0.05080607 -0.03739338
    #                  -0.03096389  0.21181234  0.0114693]
    s_numerical = np.array([0.03113369, -0.05080607, -0.03739338,
                            -0.03096389, 0.21181234, 0.0114693])
    # s_numerical = fe.calc.calculate_numerical_stress(fe, 1e-5)
    s_err = s_numerical - s_analytical

    parprint('Analytical stress:\n', s_analytical)
    parprint('Numerical stress:\n', s_numerical)
    parprint('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-4)
