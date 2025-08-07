from my_new_prepare_01 import *

atoms, calc = prepare_Al_fcc()

# This is for debugging DFT calculation object
#from my_gpaw25.new.calculation import DFTCalculation
#calc_dft = DFTCalculation.from_parameters(
#    atoms, calc.params, calc.comm, calc.log
#)

print("---- DEBUG DFTCalculation.from_parameters ----")

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
# XXX SetupData is initialized here?
# kpoints are also initialized?

# Build basis_set (?) Is this relevant for PW basis?
basis_set = builder.create_basis_set()

density = builder.density_from_superposition(basis_set)
density.normalize()

# The SCF-loop has a Hamiltonian that has an fft-plan that is
# cached for later use, so best to create the SCF-loop first
# FIX this!
scf_loop = builder.create_scf_loop() # probably not really needed here

pot_calc = builder.create_potential_calculator()

# some energies are computed here?
potential, _ = pot_calc.calculate_without_orbitals(
    density, kpt_band_comm=builder.communicators['D']
)

ibzwfs = builder.create_ibz_wave_functions(
    basis_set, potential, log=log)

if ibzwfs.wfs_qs[0][0]._eig_n is not None:
    ibzwfs.calculate_occs(scf_loop.occ_calc)

print("---- END DEBUG DFTCalculation.from_parameters ----")

