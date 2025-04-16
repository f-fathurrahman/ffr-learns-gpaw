import numpy as np
import pytest
from ase.build import bulk

from my_gpaw import GPAW, PW, Mixer
from my_gpaw.mpi import world


@pytest.mark.stress
def test_pw_si_stress(in_tmp_dir):
    xc = 'PBE'
    si = bulk('Si')
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(1, 1, 2),  # Run (1, 1, 2) to avoid gamma pt code
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   txt='si_stress.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    si.get_potential_energy()

    # Trigger nasty bug (fixed in !486):
    si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1
    
    # Compute error in stress as numerical - analytical
    s_analytical = si.get_stress()
    s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)
    s_err = s_numerical - s_analytical

    assert np.all(abs(s_err) < 1e-4)
