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


bulk.calc = calc
energy = bulk.get_potential_energy()
