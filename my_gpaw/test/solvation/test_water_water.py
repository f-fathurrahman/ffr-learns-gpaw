import pytest
from my_gpaw import GPAW, restart
from my_gpaw.cluster import Cluster
from my_gpaw.test import equal
from ase.build import molecule
from ase.units import mol, kcal
from my_gpaw.solvation import SolvationGPAW, get_HW14_water_kwargs


@pytest.fixture
def parameters():
    params = {
        'xc': 'PBE',
        'h': 0.24,
        'convergence': {
            'energy': 0.05 / 8.,
            'density': 10.,
            'eigenstates': 10.}}
    return params


@pytest.fixture
def H2O(parameters):
    vac = 4.0

    atoms = Cluster(molecule('H2O'))
    atoms.minimal_box(vac, parameters['h'])

    kwargs = get_HW14_water_kwargs()
    kwargs.update(parameters)
    atoms.calc = SolvationGPAW(**kwargs)
    atoms.get_potential_energy()

    return atoms


def test_solvation_water_water(H2O, parameters):
    SKIP_VAC_CALC = True

    if not SKIP_VAC_CALC:
        atoms = H2O.copy()
        atoms.calc = GPAW(**parameters)
        Evac = atoms.get_potential_energy()
    else:
        # h=0.24, vac=4.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
        Evac = -14.857414548

    Ewater = H2O.get_potential_energy()
    Eelwater = H2O.calc.get_electrostatic_energy()
    Esurfwater = H2O.calc.get_solvation_interaction_energy('surf')
    H2O.get_forces()
    DGSol = (Ewater - Evac) / (kcal / mol)
    print('Delta Gsol: %s kcal / mol' % DGSol)

    equal(DGSol, -6.3, 2.)
    equal(Ewater, Eelwater + Esurfwater, 1e-14)


def test_read(H2O, in_tmp_dir):
    """Read and check some basic properties"""
    fname = 'solvation.gpw'
    H2O.calc.write(fname)
    atoms, calc = restart(fname, Class=SolvationGPAW, txt='-')

    for method in ['get_potential_energy',
                   'get_eigenvalues', 'get_occupation_numbers']:
        assert getattr(calc, method)() == pytest.approx(
            getattr(H2O.calc, method)())

    calc.calculate(atoms)

    for method in ['get_potential_energy',
                   'get_eigenvalues', 'get_occupation_numbers']:
        assert getattr(calc, method)() == pytest.approx(
            getattr(H2O.calc, method)())
