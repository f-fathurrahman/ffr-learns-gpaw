from ase import Atoms
from my_gpaw import GPAW, PW

name = "Al-fcc"
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
print("Pass here 21")

atoms.calc = calc

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
    calc.scf.iterate_eigensolver(calc.wfs, calc.hamiltonian, calc.density)

    calc.scf.check_convergence(
        calc.density, calc.hamiltonian, calc.wfs, calc.log, calc.call_observers)

    converged = (calc.scf.converged and
                    calc.scf.niter >= calc.scf.niter_fixdensity)
    if converged:
        calc.scf.do_if_converged(calc.wfs, calc.hamiltonian, calc.density, calc.log)
        break

    calc.scf.update_ham_and_dens(calc.wfs, calc.hamiltonian, calc.density)
    calc.scf.niter += 1

# Don't fix the density in the next step.
calc.scf.niter_fixdensity = 0

if not converged:
    calc.scf.not_converged(calc.dens, calc.ham, calc.wfs, calc.log)


print("\n---- Script ended normally ----")