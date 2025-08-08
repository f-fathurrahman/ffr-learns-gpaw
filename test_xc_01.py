from my_new_prepare_01 import *

atoms, calc = prepare_Al_fcc()

# Get several variables that are originally arguments
params = calc.params
comm = calc.comm
log = calc.log
builder = None

# Check coordinates
from my_gpaw25.utilities import check_atoms_too_close
from my_gpaw25.utilities import check_atoms_too_close_to_boundary
check_atoms_too_close(atoms)
check_atoms_too_close_to_boundary(atoms)

from my_gpaw25.new.builder import builder as create_builder
builder = builder or create_builder(atoms, params, log.comm)
basis_set = builder.create_basis_set()
#
density = builder.density_from_superposition(basis_set)
density.normalize()
# density has its own class
scf_loop = builder.create_scf_loop() # probably not really needed here
pot_calc = builder.create_potential_calculator()


#nt_sr, nt0_g, taut_sr, e_xc, vxct_sr, dedtaut_sr = (
#    pot_calc._interpolate_and_calculate_xc(pot_calc.xc, density)
#)
# for non metaGGA taut_sr and dedtaut_sr are None
# TODO: plot the quantities? (3d quantities)

# XXX why we need to interpolate?
nt_sr, nt0_g = pot_calc._interpolate_density(density.nt_sR)
# size of nt_sr is twice of density.nt_sR i

if density.taut_sR is not None:
    taut_sr = pot_calc.interpolate(density.taut_sR)
else:
    taut_sr = None
e_xc, vxct_sr, dedtaut_sr = pot_calc.xc.calculate(nt_sr, taut_sr)
# e_xc: Float64
# vxct_sr: UGArray