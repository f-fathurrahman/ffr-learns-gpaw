from my_gpaw import GPAW
from my_gpaw.cluster import Cluster
from my_gpaw.test import equal
from ase.build import molecule
from ase.units import mol, kcal
from my_gpaw.directmin.etdm import ETDM
from my_gpaw.solvation import SolvationGPAW, get_HW14_water_kwargs


def test_solvation_water_water_etdm_lcao():
    SKIP_VAC_CALC = True

    h = 0.24
    vac = 4.0
    convergence = {
        'energy': 0.05 / 8.,
        'density': 10.,
        'eigenstates': 10.,
    }

    atoms = Cluster(molecule('H2O'))
    atoms.minimal_box(vac, h)

    if not SKIP_VAC_CALC:
        atoms.calc = GPAW(mode='lcao', xc='PBE', h=h, basis='dzp',
                          occupations={'name': 'fixed-uniform'},
                          eigensolver='etdm',
                          mixer={'backend': 'no-mixing'},
                          nbands='nao', symmetry='off',
                          convergence=convergence)
        Evac = atoms.get_potential_energy()
        print(Evac)
    else:
        # h=0.24, vac=4.0, setups: 0.9.20000, convergence: only energy 0.05 / 8
        Evac = -12.68228003345474

    atoms.calc = SolvationGPAW(mode='lcao', xc='PBE', h=h, basis='dzp',
                               occupations={'name': 'fixed-uniform'},
                               eigensolver=ETDM(
                                   linesearch_algo={'name': 'max-step'}),
                               mixer={'backend': 'no-mixing'},
                               nbands='nao', symmetry='off',
                               convergence=convergence,
                               **get_HW14_water_kwargs())
    Ewater = atoms.get_potential_energy()
    Eelwater = atoms.calc.get_electrostatic_energy()
    Esurfwater = atoms.calc.get_solvation_interaction_energy('surf')
    DGSol = (Ewater - Evac) / (kcal / mol)
    print('Delta Gsol: %s kcal / mol' % DGSol)

    equal(DGSol, -6.3, 2.)
    equal(Ewater, Eelwater + Esurfwater, 1e-14)
