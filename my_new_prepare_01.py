from my_gpaw25.new.ase_interface import GPAW
from my_gpaw25 import PW
from ase import Atoms
from ase.build import molecule
from ase.units import Ha

def prepare_Al_fcc():
    a = 4.05
    b = a/2
    atoms = Atoms("Al",
        cell=[[0, b, b],
            [b, 0, b],
            [b, b, 0]],
    pbc=True)
    k = 4
    calc = GPAW(
        mode=PW(300),
        kpts=(k,k,k),
        txt="-"
    )
    atoms.calc = calc
    return atoms, calc


def prepare_H2O():
    atoms = molecule("H2O", vacuum=3.0)
    ecutwfc = 15*Ha
    calc = GPAW(mode=PW(ecutwfc), txt="-")
    atoms.calc = calc
    return atoms, calc
