from ase.units import Ha
from my_new_prepare_01 import *

#atoms, calc = prepare_Al_fcc()
atoms, calc = prepare_H2O()
energy = atoms.get_potential_energy()
