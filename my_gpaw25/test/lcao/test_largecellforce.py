# This test calculates the force on the atoms in a hydrogen chain, comparing
# the results to finite-difference values.
#
# The reference result is obtained from an FD calculation, which can be rerun
# by setting the fd boolean below.
#
# The purpose is to test domain decomposition with large cells.  The basis
# functions of one atom are defined to not overlap the rightmost domain
# for z-decompositions of two or more slices.  This will change the set
# of atomic indices stored in BasisFunctions objects and other things
#
# This test also ensures that lcao forces are tested with non-pbc.
import numpy as np
from numpy import array
from ase import Atoms
from my_gpaw25 import GPAW
from my_gpaw25.atom.basis import BasisMaker


def test_lcao_largecellforce(gpaw_new):
    hbasis = BasisMaker('H').generate(1, 0, energysplit=1.8, tailnorm=0.03**.5)
    basis = {'H': hbasis}

    atom = Atoms('H')
    atom.center(vacuum=.8)
    system = atom.repeat((1, 1, 4))

    system.center(vacuum=2.5)

    calc = GPAW(h=0.23,
                mode='lcao',
                basis=basis,
                convergence={'density': 1e-4, 'energy': 1e-7})

    system.calc = calc

    F_ac = system.get_forces()

    # Check that rightmost domain is in fact outside range of basis functions
    from my_gpaw25.mpi import rank, size
    if rank == 0 and size > 1:
        if gpaw_new:
            basis = calc.dft.scf_loop.hamiltonian.basis
        else:
            basis = calc.wfs.basis_functions
        assert len(basis.atom_indices) < len(system)

    fd = 0

    # Values taken from FD calculation below
    # (Symmetry means only z-component may be nonzero)
    ref = array([[0, 0, 4.616841597363752],
                 [0, 0, -2.7315136482540803],
                 [0, 0, 2.7315116638237935],
                 [0, 0, -4.616840606709416]])

    if fd:
        from my_gpaw25.test import calculate_numerical_forces
        ref = calculate_numerical_forces(system, 0.002, icarts=[2])
        print('Calced')
        print(F_ac)
        print('FD')
        print(ref)
        print(repr(ref))

    err = np.abs(F_ac - ref).max()
    print('Ref')
    print(ref)
    print('Calculated')
    print(F_ac)
    print('Max error', err)
    assert err < 6e-4
