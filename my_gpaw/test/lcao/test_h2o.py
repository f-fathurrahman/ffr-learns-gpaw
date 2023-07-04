import pytest
from ase import Atoms

from gpaw import GPAW, LCAO
from gpaw.poisson import FDPoissonSolver
from gpaw.test import equal


@pytest.mark.later  # basis set cutoff??
def test_lcao_h2o():
    a = 6.0
    b = a / 2
    mol = Atoms('OHH',
                [(b, b, 0.1219 + b),
                 (b, 0.7633 + b, -0.4876 + b),
                 (b, -0.7633 + b, -0.4876 + b)],
                pbc=False, cell=[a, a, a])
    calc = GPAW(gpts=(32, 32, 32),
                nbands=4,
                mode='lcao',
                poissonsolver=FDPoissonSolver())
    mol.calc = calc
    e = mol.get_potential_energy()
    niter = calc.get_number_of_iterations()

    equal(e, -10.271, 2e-3)
    equal(niter, 8, 1)

    # Check that complex wave functions are allowed with
    # gamma point calculations
    calc = GPAW(gpts=(32, 32, 32),
                nbands=4,
                mode=LCAO(force_complex_dtype=True),
                poissonsolver=FDPoissonSolver())
    mol.calc = calc
    ec = mol.get_potential_energy()
    equal(e, ec, 1e-5)
