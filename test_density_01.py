from my_new_prepare_01 import *

atoms, calc = prepare_Al_fcc()
#atoms, calc = prepare_Al_fcc(setups="hgh")
#atoms, calc = prepare_H2O()
#atoms, calc = prepare_H2O(setups="hgh")

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
from my_gpaw25.new.density import Density
#density = builder.density_from_superposition(basis_set)
density = Density.from_superposition(
            grid=builder.grid,
            nct_aX=builder.get_pseudo_core_densities(),
            tauct_aX=builder.get_pseudo_core_ked(),
            atomdist=builder.atomdist,
            setups=builder.setups,
            basis_set=basis_set,
            magmom_av=builder.initial_magmom_av,
            ncomponents=builder.ncomponents,
            charge=builder.params.charge,
            hund=builder.params.hund,
            mgga=builder.xc.type == "MGGA")
print("integrated chg (before normalize) = ", density.nt_sR.integrate())
density.normalize()
print("integrated chg (after normalize) = ", density.nt_sR.integrate())

from math import sqrt, pi

# Test normalize
comp_charge = 0.0
xp = density.D_asii.layout.xp
for a, D_sii in density.D_asii.items():
    comp_charge += xp.einsum('sij, ij ->',
                                D_sii[:density.ndensities].real,
                                density.delta_aiiL[a][:, :, 0])
    comp_charge += density.delta0_a[a]
# comp_charge could be cupy.ndarray:
comp_charge = float(comp_charge) * sqrt(4 * pi)
comp_charge = density.nt_sR.desc.comm.sum_scalar(comp_charge)
charge = comp_charge + density.charge
pseudo_charge = density.nt_sR[:density.ndensities].integrate().sum()
if pseudo_charge != 0.0:
    x = -charge / pseudo_charge
    #self.nt_sR.data *= x
print("x = ", x)