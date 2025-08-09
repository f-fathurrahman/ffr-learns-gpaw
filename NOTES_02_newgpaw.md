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

TODO: learn some objects and methods in
`calc_dft` and `calc_dft.scf_loop`

Some methods in calc_dft are quite important (general for both SCF loop and direct minimization)

Make a version of SCF iteration that does not use `yield`

# PWDesc and PWArray

Examples ...

# UGArray

Examples ...


# PWAtomCenteredFunctions

Example, output from `builder.get_pseudo_core_densities`

# Computing total energy

```text
New.ASECalculator.calculate_property
  New.ASECalculator.iconverge
    
    New.ASECalculator.create_new_calculation
      New.DFTCalculation.from_parameters
    
    New.DFTCalculation.iconverge
      New.SCFLoop.iterate
       ....
```

# Computing energies

Need two parameters: potential and ibzwfs
Potential energies: `potential.energies` (a data field in `Potential`)
Example output (as `dict`):
```
{'coulomb': np.float64(-0.2451148800157461),
 'zero': np.float64(0.05500662939980305),
 'xc': np.float64(-0.22341431839757964),
 'stress': np.float64(3.9927748584207907),
 'external': np.float64(0.0),
 'kinetic': np.float64(-0.017809744647594328),
 'spinorbit': np.float64(0.0)}
```
Why there is kinetic energy here?

From `ibzwfs`:
```
{'band': 0.30611204200136466,
 'entropy': -0.0009564838067774605,
 'extrapolation': 0.00047824190338873027}
```

# Computing potentials

For potentials, we have `Potential` class.

In PW calculation this is handled by `PlaneWavePotentialCalculator`.

Some important methods:
- `calculate_pseudo_potential`: I think this contains many PAW-specific stuffs.


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