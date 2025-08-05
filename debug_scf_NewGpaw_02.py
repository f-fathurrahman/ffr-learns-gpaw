from my_gpaw25.new.ase_interface import GPAW
#from my_gpaw25 import GPAW # old
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

atoms, calc = prepare_Al_fcc()

# Next step can be:

# call atoms.get_potential_energy(): get the energy directly
#energy = atoms.get_potential_energy()

# Do the SCF iterations: energy is not directly available
#for _ in calc.iconverge(atoms):
#    pass
# calc is a Calculator object


# This is for debugging DFT calculation object
from my_gpaw25.new.calculation import DFTCalculation
calc_dft = DFTCalculation.from_parameters(
    atoms, calc.params, calc.comm, calc.log
)
# This will converge the actual calculation, using yield
#for _ in calc_dft.iconverge():
#    pass

"""
# Call via scf_loop.iterate
# Calling this will return a Generator object
ret_obj = calc_dft.scf_loop.iterate(
    calc_dft.ibzwfs,
    calc_dft.density,
    calc_dft.potential,
    calc_dft.pot_calc,
    maxiter=2,
    calculate_forces=True,
    log=calc_dft.log
)

# The actual loop need to be executed in loop ?
for _ in ret_obj:
    pass
"""

