import numpy as np

from my_new_prepare_01 import *

from my_gpaw25.core.atom_centered_functions import AtomArraysLayout
from my_gpaw25.utilities import unpack_density, unpack_hermitian
from my_gpaw25.typing import Vector
from my_gpaw25.new import zips

def my_atomic_occupation_numbers(
    setup, magmom_v: Vector,
    ncomponents: int, hund: bool = False, charge: float = 0.0):
    #
    M = np.linalg.norm(magmom_v)
    nspins = min(ncomponents, 2)
    f_si = setup.calculate_initial_occupation_numbers(
        M, hund, charge=charge, nspins=nspins)

    if ncomponents == 1:
        pass
    elif ncomponents == 2:
        if magmom_v[2] < 0:
            f_si = f_si[::-1].copy()
    else:
        f_i = f_si.sum(0)
        fm_i = f_si[0] - f_si[1]
        f_si = np.zeros((4, len(f_i)))
        f_si[0] = f_i
        if M > 0:
            f_si[1:] = np.asarray(magmom_v)[:, np.newaxis] / M * fm_i

    return f_si



def my_dens_from_superposition(
    grid, nct_aX, tauct_aX, atomdist, setups,
    basis_set, magmom_av, ncomponents, charge=0.0, hund=False, mgga=False):
    #
    nt_sR = grid.zeros(ncomponents)
    atom_array_layout = AtomArraysLayout(
        [(setup.ni, setup.ni) for setup in setups],
        atomdist=atomdist, dtype=float if ncomponents < 4 else complex)
    D_asii = atom_array_layout.empty(ncomponents)
    f_asi = {
        a: my_atomic_occupation_numbers(setup, magmom_v, ncomponents,hund, charge / len(setups))
                for a, (setup, magmom_v) in enumerate(zips(setups, magmom_av))
    }
    #
    basis_set.add_to_density(nt_sR.data, f_asi)
    for a, D_sii in D_asii.items():
        D_sii[:] = unpack_density(
            setups[a].initialize_density_matrix(f_asi[a]))

    xp = nct_aX.xp
    nt_sR = nt_sR.to_xp(xp)
    density = my_dens_from_data_and_setups(nt_sR,
                                        None,
                                        D_asii.to_xp(xp),
                                        charge,
                                        setups,
                                        nct_aX,
                                        tauct_aX)
    ndensities = ncomponents % 3
    density.nt_sR.data[:ndensities] += density.nct_R.data
    if mgga:
        density.taut_sR = nt_sR.new()
        density.taut_sR.data[:] = density.tauct_R.data
    return density


def my_dens_from_data_and_setups(
    nt_sR, taut_sR, D_asii,
    charge, setups, nct_aX, tauct_aX):
    #
    xp = nt_sR.xp
    return Density(nt_sR,
                taut_sR,
                D_asii,
                charge,
                [xp.asarray(setup.Delta_iiL) for setup in setups],
                [setup.Delta0 for setup in setups],
                [unpack_hermitian(setup.N0_p) for setup in setups],
                [setup.n_j for setup in setups],
                [setup.l_j for setup in setups],
                nct_aX,
                tauct_aX)






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
#density = Density.from_superposition(
density = my_dens_from_superposition(
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