Only need the following changes:
```python
#from my_gpaw25 import GPAW # old
from my_gpaw25.new.ase_interface import GPAW
```

Setup calculation:

gpaw.new.calculation class, `from_parameter` method.

Many important objects seem to be defined in this class.

For each objects, learn how to initialize it. What parameters are
needed to be constructed previously?


# Wave functions

Variable: DFTCalculation.ibzwfs

Parent class IBZWavefunction

Investigate `__str__` method to see some interesting fields.

# Potential

Create potential via `builder`
```python
pot_calc = builder.create_potential_calculator()
```

new.pw.PlaneWavePotentialCalculator

# Hamiltonian

new.pw.hamiltonian.PWHamiltonian



# Kpoints and lattice symmetries

`ibzwfs.ibz` of type `gpaw.new.brillouin.IBZ`
`ibzwfs.ibz.symmetries` of type `gpaw.new.symmetry.Symmetries`

# Basis set (need this for plane waves?)

Type: `gpaw.lfc.BasisFunctions`