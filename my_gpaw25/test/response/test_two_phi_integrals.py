import pytest
from my_gpaw25 import GPAW
from my_gpaw25.response.groundstate import ResponsePAWDataset
from my_gpaw25.response.paw import calculate_pair_density_correction
import numpy as np


@pytest.mark.response
def test_two_phi_integrals(gpw_files):
    calc = GPAW(gpw_files['bn_pw'])

    setup = calc.wfs.setups[0]
    pawdata = ResponsePAWDataset(setup)
    k_Gv = np.array([[0.0, 0.0, 0.0]])
    dO_aii = calculate_pair_density_correction(k_Gv, pawdata=pawdata)
    assert dO_aii[0] == pytest.approx(setup.dO_ii, 1e-8, 1e-7)
