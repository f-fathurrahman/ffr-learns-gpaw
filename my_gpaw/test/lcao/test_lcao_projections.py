import numpy as np
import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.lcao.projected_wannier import get_lcao_projections_HSP
from gpaw.poisson import FDPoissonSolver


@pytest.mark.later
def test_lcao_lcao_projections():
    atoms = molecule('C2H2')
    atoms.center(vacuum=3.0)
    calc = GPAW(gpts=(32, 32, 48),
                experimental={'niter_fixdensity': 2},
                poissonsolver=FDPoissonSolver(),
                eigensolver='rmm-diis')
    atoms.calc = calc
    atoms.get_potential_energy()

    V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
        calc, bfs=None, spin=0, projectionsonly=False)

    # Test H and S
    eig = sorted(np.linalg.eigvals(np.linalg.solve(S_qMM[0], H_qMM[0])).real)
    eig_ref = np.array([-17.879394403125634, -13.248793622886552,
                        -11.431247684217933, -7.125771721594344,
                        -7.125771721594252, 0.5927469818425286,
                        0.5927469818425768, 3.925124575829719,
                        7.451028687071511, 26.734351930654853])
    print(eig)
    assert np.allclose(eig, eig_ref)
