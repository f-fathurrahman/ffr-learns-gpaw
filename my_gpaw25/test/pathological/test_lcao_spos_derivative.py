import numpy as np
import pytest
from ase.build import bulk

from my_gpaw25 import GPAW, FermiDirac, Mixer


@pytest.mark.old_gpaw_only
def test_pathological_lcao_spos_derivative(in_tmp_dir):
    atoms = bulk('Si', 'diamond', a=5.4834322363595565)
    atoms *= (3, 3, 3)
    atoms.calc = GPAW(
        gpts=(32, 32, 32),
        mixer=Mixer(0.5, 5, 50.0),
        txt='grumble.txt',
        kpts=(2, 1, 1),
        mode='lcao',
        basis='sz(dzp)',
        xc='oldLDA',
        occupations=FermiDirac(0.01))
    f = atoms.get_forces()
    fmax = np.abs(f).max()
    print('maxforce', fmax)
    assert fmax < 0.05  # 0.03 normally, 2.7 with bug
    # fmax can be converged much closer to 0 with better grid/SCF convergence.
