from ase.build import molecule
from my_gpaw25 import GPAW, PW
from my_gpaw25.mpi import world


def test_pw_h(in_tmp_dir):
    a = molecule('H', pbc=1)
    a.center(vacuum=2)

    comm = world.new_communicator([world.rank])
    e0 = 0.0
    a.calc = GPAW(mode=PW(250),
                  communicator=comm,
                  txt=None)
    e0 = a.get_potential_energy()
    e0 = world.sum_scalar(e0) / world.size

    a.calc = GPAW(mode=PW(250),
                  eigensolver='rmm-diis',
                  basis='szp(dzp)',
                  txt='%d.txt' % world.size)
    e = a.get_potential_energy()
    f = a.get_forces()
    assert abs(e - e0) < 7e-5, abs(e - e0)
    assert abs(f).max() < 1e-10, abs(f).max()
