```python
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
```

Constructing a `Calculator` object like this seems to only allocate various common setup
variables or options.
The real works start when `get_potential_energy` method is called.
```python
bulk.calc = calc
calc.get_potential_energy()
```

The method `get_potential_energy` is inherited from `ase.Calculator` object.
It calls `get_property` which calls `calculate`.
In `GPAW` calculator it calls `icalculate`.
In `icalculate` interesting things happen in `initialize` method.

`GPAW` class also has `self.scf` object which seems to be responsible
for the SCF calculation.



ReciprocalSpaceHamiltonian

gpaw.scf.irun