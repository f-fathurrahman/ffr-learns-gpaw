import pytest
from ase.build import molecule

from my_gpaw25 import GPAW
from my_gpaw25.utilities.adjust_cell import adjust_cell
from my_gpaw25.analyse.overlap import Overlap
import my_gpaw25.solvation as solv
from my_gpaw25.lrtddft import LrTDDFT
from my_gpaw25.poisson import PoissonSolver


@pytest.mark.skip(reason='TODO')
def test_solvation_overlap():
    """Check whether LrTDDFT in solvation works"""

    h = 0.4
    box = 2
    nbands = 2

    H2 = molecule('H2')
    adjust_cell(H2, box, h)

    c1 = GPAW(mode='fd', h=h, txt=None, nbands=nbands)
    c1.calculate(H2)

    c2 = solv.SolvationGPAW(mode='fd',
                            h=h,
                            txt=None,
                            nbands=nbands + 1,
                            **solv.get_HW14_water_kwargs())
    c2.calculate(H2)
    for poiss in [None, PoissonSolver(nn=c2.hamiltonian.poisson.nn)]:
        lr = LrTDDFT(c2, poisson=poiss)
        print(lr)
    print(Overlap(c1).pseudo(c2))
