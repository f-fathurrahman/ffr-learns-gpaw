import pytest
from ase import Atoms

from my_gpaw25 import GPAW
from my_gpaw25.external import ConstantPotential


@pytest.mark.old_gpaw_only
def test_ext_potential_external():
    sc = 2.9
    R = 0.7  # approx. experimental bond length
    R = 1.0
    a = 2 * sc
    c = 3 * sc
    H2 = Atoms('HH', [(a / 2, a / 2, (c - R) / 2),
                      (a / 2, a / 2, (c + R) / 2)],
               cell=(a, a, c),
               pbc=False)

    txt = None

    convergence = {'eigenstates': 1.e-4 * 40 * 1.5**3,
                   'density': 1.e-2,
                   'energy': 0.1}

    # without potential
    if True:
        if txt:
            print('\n################## no potential')
        c00 = GPAW(mode='fd', h=0.3, nbands=-1,
                   convergence=convergence,
                   txt=txt)
        c00.calculate(H2)
        c00.get_eigenvalues()

    # 0 potential
    if True:
        if txt:
            print('\n################## 0 potential')
        cp0 = ConstantPotential(0.0)
        c01 = GPAW(mode='fd', h=0.3, nbands=-2, external=cp0,
                   convergence=convergence,
                   txt=txt)
        c01.calculate(H2)

    # 1 potential
    if True:
        if txt:
            print('################## 1 potential')
        cp1 = ConstantPotential(-1.0)
        c1 = GPAW(mode='fd', h=0.3, nbands=-2, external=cp1,
                  convergence=convergence,
                  txt=txt)
        c1.calculate(H2)

    for i in range(c00.get_number_of_bands()):
        f00 = c00.get_occupation_numbers()[i]
        if f00 > 0.01:
            e00 = c00.get_eigenvalues()[i]
            e1 = c1.get_eigenvalues()[i]
            print('Eigenvalues no pot, expected, error=',
                  e00, e1 + 1, e00 - e1 - 1)
            assert e00 == pytest.approx(e1 + 1., abs=0.008)

    E_c00 = c00.get_potential_energy()
    E_c1 = c1.get_potential_energy()
    DeltaE = E_c00 - E_c1
    assert DeltaE == pytest.approx(0, abs=0.002)
