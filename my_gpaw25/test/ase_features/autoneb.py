# Run with even number of cores

import pytest
from my_gpaw25 import GPAW, Mixer, PoissonSolver
import ase.parallel as mpi
from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.autoneb import AutoNEB


@pytest.mark.skip(reason='TODO')
def test_autoneb():
    from ase.mep.neb import NEBTools
    size = mpi.world.size
    rank = mpi.world.rank

    slab = fcc211('Ag', size=(3, 2, 1), vacuum=3.0)
    add_adsorbate(slab, 'Ag', 0.5, (-0.1, 2.7))

    slab.set_constraint(FixAtoms(range(6)))
    slab.pbc = 1

    def getcalc(**kwargs):
        kwargs1 = dict(xc='oldLDA',
                       gpts=(32, 24, 32),
                       setups={'Ag': '11'},
                       # h=0.24,
                       poissonsolver=PoissonSolver(relax='GS'),
                       mixer=Mixer(0.5, 5, 50.0),
                       mode='lcao',
                       basis='sz(dzp)')
        kwargs1.update(kwargs)
        return GPAW(**kwargs1)

    calc = getcalc()

    slab.calc = calc

    qn = QuasiNewton(slab, trajectory='neb000.traj')
    qn.run(fmax=0.05)

    slab[-1].x += slab.get_cell()[0, 0]
    slab[-1].y += 2.8

    qn = QuasiNewton(slab, trajectory='neb001.traj')
    qn.run(fmax=0.05)

    def attach_calculators(images):
        nim = len(images)
        n = size // nim  # number of cpu's per image
        j = rank // n  # image number
        assert nim * n == size

        for i in range(nim):
            ranks = range(i * n, (i + 1) * n)
            if rank in ranks:
                calc = getcalc(txt='neb%d.txt' % j,
                               communicator=ranks)
                images[i].calc = calc

    autoneb = AutoNEB(attach_calculators,
                      prefix='neb',
                      n_simul=2,
                      parallel=True,
                      climb=True,
                      n_max=5,
                      optimizer='FIRE',
                      fmax=0.05,
                      k=0.5,
                      maxsteps=[25, 1000])
    autoneb.run()

    nebtools = NEBTools(autoneb.all_images)
    barrier, delta_e = nebtools.get_barrier()
    print('barrier', barrier)
    ref = 0.74051020956857272  # 1.484 <- with better parameters
    err = abs(barrier - ref)
    assert err < 1e-3, 'barrier={}, expected={}, err={}'.format(barrier,
                                                                ref, err)
