import pytest
from gpaw import GPAW
from gpaw.response.paw import two_phi_planewave_integrals
import numpy as np


@pytest.mark.response
def test_two_phi_integrals(gpw_files):
    calc = GPAW(gpw_files['bn_pw_wfs'])
    
    setup = calc.wfs.setups[0]
    k_Gv = np.array([[0.0, 0.0, 0.0]])
    dO_ii = two_phi_planewave_integrals(k_Gv, pawdata=setup)
    ni = len(setup.dO_ii)
    assert dO_ii.reshape(ni, ni) == pytest.approx(setup.dO_ii, 1e-8, 1e-7)
