from ase import Atoms

from my_gpaw import GPAW, FermiDirac
from my_gpaw.test import equal


def test_spin_spinFe3plus():
    h = 0.4
    q = 3

    s = Atoms('Fe')
    s.center(vacuum=2.5)
    convergence = {'eigenstates': 0.01, 'density': 0.1, 'energy': 0.1}

    # use Hunds rules

    c = GPAW(charge=q, h=h, nbands=5,
             hund=True,
             eigensolver='rmm-diis',
             occupations=FermiDirac(width=0.1),
             convergence=convergence)
    s.calc = c
    equal(s.get_magnetic_moment(), 5, 0.1)

    # set magnetic moment

    mm = [5]
    s.set_initial_magnetic_moments(mm)
    c = GPAW(charge=q, h=h, nbands=5,
             occupations=FermiDirac(width=0.1, fixmagmom=True),
             convergence=convergence)
    s.calc = c
    equal(s.get_magnetic_moment(), 5, 0.1)
