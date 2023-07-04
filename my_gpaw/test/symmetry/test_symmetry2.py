"""This script asserts that the chi's obtained
from GS calculations using symmetries
and GS calculations not using symmetries return
the same results. Tests that the chi's are element-wise
equal to a tolerance of 1e-10.
"""

import pytest
import numpy as np

from gpaw.response.chi0 import Chi0


@pytest.mark.response
@pytest.mark.slow
def test_response_symmetry(gpw_files):
    data_s = []
    for name in ['ti2o4_pw_nosym_wfs', 'ti2o4_pw_wfs']:
        X = Chi0(gpw_files[name])
        chi_t = X.calculate([1. / 4, 0, 0])
        data_s.append((chi_t.chi0_WgG, chi_t.chi0_WxvG, chi_t.chi0_Wvv))

    msg = 'Difference in Chi when turning off symmetries!'

    while len(data_s):
        data1 = data_s.pop()
        for data2 in data_s:
            for dat1, dat2 in zip(data1, data2):
                if dat1 is not None:
                    assert (np.abs(dat1 - dat2).max() ==
                            pytest.approx(0, abs=0.001)), msg
