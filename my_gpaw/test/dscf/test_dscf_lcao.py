import pytest
from ase.build import molecule
from my_gpaw import GPAW
import gpaw.dscf as dscf
from my_gpaw.test import equal

# Ground state calculation


@pytest.mark.dscf
def test_dscf_dscf_lcao():
    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=8,
                h=0.2,
                xc='PBE',
                spinpol=True,
                convergence={'energy': 100,
                             'density': 1e-3,
                             'bands': -1})

    CO = molecule('CO')
    CO.center(vacuum=3)
    CO.calc = calc

    E_gs = CO.get_potential_energy()

    # Excited state calculation

    calc_es = GPAW(mode='lcao',
                   basis='dzp',
                   nbands=8,
                   h=0.2,
                   symmetry='off',
                   xc='PBE',
                   spinpol=True,
                   convergence={'energy': 100,
                                'density': 1e-3,
                                'bands': -1})

    CO.calc = calc_es
    lumo = dscf.MolecularOrbital(calc,
                                 weights={0: [0, 0, 0, 1], 1: [0, 0, 0, -1]})
    dscf.dscf_calculation(calc_es, [[1.0, lumo, 1]], CO)

    E_es = CO.get_potential_energy()
    dE = E_es - E_gs
    print(dE)
    equal(dE, 5.7595110076, 0.011)
