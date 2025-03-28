from ase import Atoms
from my_gpaw import GPAW, PW

name = "Al-fcc"
a = 4.05
b = a/2

bulk = Atoms("Al",
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

bulk.calc = calc
for _ in calc.icalculate(bulk):
    pass
#calc.icalculate(atoms=bulk, properties=['energy'])

print("pass here 23")