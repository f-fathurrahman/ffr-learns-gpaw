from ase import Atoms
from gpaw import GPAW, PoissonSolver
from gpaw.test import equal
from ase.units import Bohr, Hartree


def test_xc_revPBE():
    a = 7.5 * Bohr
    n = 16
    atoms = Atoms('He', [(0.0, 0.0, 0.0)], cell=(a, a, a), pbc=True)
    params = dict(gpts=(n, n, n), nbands=1, xc={'name': 'PBE', 'stencil': 1},
                  poissonsolver=PoissonSolver('fd'))
    atoms.calc = GPAW(**params)
    e1 = atoms.get_potential_energy()
    e1a = atoms.calc.get_reference_energy()
    params['xc'] = {'name': 'revPBE', 'stencil': 1}
    atoms.calc = GPAW(**params)
    e2 = atoms.get_potential_energy()
    e2a = atoms.calc.get_reference_energy()

    equal(e1a, -2.893 * Hartree, 8e-3)
    equal(e2a, -2.908 * Hartree, 9e-3)
    equal(e1, e2, 4e-3)

    energy_tolerance = 0.0005
    equal(e1, -0.0790449962, energy_tolerance)
    equal(e2, -0.08147563, energy_tolerance)
