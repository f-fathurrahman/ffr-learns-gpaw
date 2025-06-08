from ase import Atoms
from ase.build import molecule
from ase.units import Ha
from my_gpaw import GPAW, PW

def prepare_Al_fcc():
    name = "Al-fcc"
    a = 4.05
    b = a/2
    atoms = Atoms(name,
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


#atoms, calc = prepare_Al_fcc()
atoms, calc = prepare_H2O()



# icalculate must be called in a `for` loop? because it is a generator
#calc.icalculate(bulk)
#for _ in calc.icalculate(bulk):
#    pass

# Drastic changes:
calc.wfs = None
calc.density = None
calc.hamiltonian = None
calc.scf = None

from my_gpaw_initialize import my_gpaw_initialize
#calc.initialize(atoms)
my_gpaw_initialize(calc, atoms)

# Need this for projections (?)
calc.set_positions(atoms)

#
# from scf.irun
#
calc.scf.eigensolver_used = getattr(calc.wfs.eigensolver, "name", None)
calc.scf.check_eigensolver_state(calc.wfs, calc.hamiltonian, calc.density)
calc.scf.niter = 1
converged = False


while calc.scf.niter <= calc.scf.maxiter:
    
    print(f"\n<SCF_ITERATION> BEGIN SCF ITER = {calc.scf.niter}\n")

    # FIXME: change to plain function call
    calc.scf.iterate_eigensolver(calc.wfs, calc.hamiltonian, calc.density)

    calc.scf.check_convergence(
        calc.density, calc.hamiltonian, calc.wfs, calc.log, calc.call_observers)

    converged = (calc.scf.converged and
                    calc.scf.niter >= calc.scf.niter_fixdensity)
    if converged:
        calc.scf.do_if_converged(calc.wfs, calc.hamiltonian, calc.density, calc.log)
        break

    calc.scf.update_ham_and_dens(calc.wfs, calc.hamiltonian, calc.density)

    print(f"\n</SCF_ITERATION> END SCF ITER = {calc.scf.niter}\n")
    calc.scf.niter += 1



# Don't fix the density in the next step.
calc.scf.niter_fixdensity = 0

if not converged:
    calc.scf.not_converged(calc.dens, calc.ham, calc.wfs, calc.log)


print("\n---- Script ended normally ----")