from my_gpaw25.new.ase_interface import GPAW
from my_gpaw25 import PW
from ase.build import molecule
from ase.units import Ha

atoms = molecule("H2O", vacuum=3.0)

ecutwfc = 15*Ha
print("ecutwfc = ", ecutwfc)

calc = GPAW(mode=PW(15*Ha), txt="-")

atoms.calc = calc
energy = atoms.get_potential_energy()

