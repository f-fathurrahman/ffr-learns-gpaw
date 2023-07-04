from types import SimpleNamespace

from my_gpaw.kpt_descriptor import KPointDescriptor
from my_gpaw.lfc import BasisFunctions
from my_gpaw.mpi import serial_comm


def create_basis(ibz,
                 nspins,
                 pbc_c,
                 grid,
                 setups,
                 dtype,
                 fracpos_ac,
                 world=serial_comm,
                 kpt_comm=serial_comm):
    kd = KPointDescriptor(ibz.bz.kpt_Kc, nspins)
    kd.set_symmetry(SimpleNamespace(pbc=pbc_c),
                    ibz.symmetries.symmetry,
                    comm=world)
    kd.set_communicator(kpt_comm)

    basis = BasisFunctions(grid._gd,
                           [setup.basis_functions_J for setup in setups],
                           kd,
                           dtype=dtype,
                           cut=True)
    basis.set_positions(fracpos_ac)
    return basis
