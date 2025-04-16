import sys

from ase import Atoms
from ase.utils.timing import Timer
import pytest

from my_gpaw import GPAW
from my_gpaw.test import equal
from my_gpaw.xc.hybrid import HybridXC


@pytest.mark.libxc
@pytest.mark.hybrids
def test_exx_coarse(in_tmp_dir):
    timer = Timer()

    loa = Atoms('Be2',
                [(0, 0, 0), (2.45, 0, 0)],
                magmoms=[0.5, 0.5],
                cell=[5.9, 4.8, 5.0])
    loa.center()

    fgl = [False, True]
    # fgl = [True, False]

    E = {}
    niter = {}
    for fg in fgl:
        if fg:
            tstr = 'Exx on fine grid'
        else:
            tstr = 'Exx on coarse grid'
        timer.start(tstr)
        loa.calc = GPAW(h=0.3,
                        eigensolver='rmm-diis',
                        xc=dict(name='PBE', stencil=1),
                        poissonsolver={'name': 'fd'},
                        nbands=4,
                        convergence={'eigenstates': 1e-4},
                        charge=-1)
        E[fg] = loa.get_potential_energy()
        loa.calc = loa.calc.new(xc=HybridXC('PBE0', stencil=1, finegrid=fg))
        E[fg] = loa.get_potential_energy()
        niter[fg] = loa.calc.get_number_of_iterations()
        timer.stop(tstr)
        if not fg:
            fname = 'exx_load.gpw'
            loa.calc.write(fname)
            calcl = GPAW(fname)
            func = calcl.parameters.xc

            assert func['name'] == 'PBE0', 'wrong name for functional'
            assert func['hybrid'] == 0.25, 'wrong factor for functional'

    timer.write(sys.stdout)

    print('Total energy on the fine grid   =', E[True])
    print('Total energy on the coarse grid =', E[False])
    equal(E[True], E[False], 0.01)

    energy_tolerance = 0.003
    equal(E[False], 6.97818, energy_tolerance)
    equal(E[True], 6.97153, energy_tolerance)
