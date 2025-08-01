from ase import Atoms
from my_gpaw25 import GPAW, FermiDirac


def test_symmetry_usesymm():
    a = 2.0
    L = 5.0

    a1 = Atoms('H', pbc=(1, 0, 0), cell=(a, L, L))
    a1.center()
    atoms = a1.repeat((4, 1, 1))

    def energy(symm={}):
        atoms.calc = GPAW(h=0.3,
                          occupations=FermiDirac(width=0.1),
                          symmetry=symm,
                          convergence=dict(energy=1e-6),
                          kpts=(3, 1, 1),
                          mode='lcao')
        return atoms.get_potential_energy()

    e1 = energy()
    e2 = energy('off')

    print(e1)
    print(e2)

    assert abs(e2 - e1) < 1e-6
