from ase.build import bulk
from gpaw import GPAW

# Part 1: Ground state calculation
atoms = bulk("Si", "diamond", a=5.431)
calc = GPAW(mode="pw", kpts=(4, 4, 4))

atoms.calc = calc
atoms.get_potential_energy()  # ground state calculation is performed
calc.write("Si_gs.gpw", "all")  # use "all" option to write wavefunction
