# {   -1,   -1}:  On entry to PDLASRT parameter number 9 had an illegal value

# works with 'sl_default': (2, 2, 32)

import pytest
from ase.build import fcc100, add_adsorbate
from my_gpaw25 import GPAW, ConvergenceError
from my_gpaw25.mpi import world
from my_gpaw25.utilities import compiled_with_sl


@pytest.mark.skip(reason='TODO')
def test_scalapack_pdlasrt_hang():
    assert world.size == 4

    slab = fcc100('Cu', size=(2, 2, 2))
    add_adsorbate(slab, 'O', 1.1, 'hollow')
    slab.center(vacuum=3.0, axis=2)

    if compiled_with_sl():
        parallel = {'domain': (1, 1, 4), 'sl_default': (2, 2, 64)}
    else:
        parallel = None

    calc = GPAW(mode='lcao',
                kpts=(2, 2, 1),
                txt='-',
                maxiter=1,
                parallel=parallel)

    slab.calc = calc
    try:
        slab.get_potential_energy()
    except ConvergenceError:
        pass
