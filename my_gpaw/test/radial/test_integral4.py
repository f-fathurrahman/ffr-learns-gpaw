import pytest
import numpy.random as ra
from my_gpaw.setup import create_setup
from my_gpaw.xc import XC


@pytest.mark.ci
def test_radial_integral4():
    ra.seed(8)
    xc = XC('LDA')
    s = create_setup('H', xc)
    ni = s.ni
    nii = ni * (ni + 1) // 2
    D_p = 0.1 * ra.random((1, nii)) + 0.2

    def f(x):
        return x

    s.xc_correction.four_phi_integrals(D_p, f)

    # Check integrals using two_phi_integrals function and finite differences:
    pass
