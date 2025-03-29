from ase import Atoms
from my_gpaw import GPAW, PW

name = "Al-fcc"
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
print("Pass here 21")

atoms.calc = calc

# icalculate must be called in a `for` loop? because it is a generator
#calc.icalculate(bulk)
#for _ in calc.icalculate(bulk):
#    pass

# Drastic changes:
calc.wfs = None
calc.density = None
calc.hamiltonian = None
calc.scf = None

from my_gpaw_initialize import my_gpaw_initialize
#calc.initialize(atoms)
my_gpaw_initialize(calc, atoms)

print("\n---- Script ended normally ----")