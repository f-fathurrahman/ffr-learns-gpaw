"""Basic test of elph/supercell/

"""
import numpy as np
import pytest

from gpaw.elph import Supercell
from gpaw.mpi import world

SUPERCELL = (2, 1, 1)

g00 = np.array([[[[-1.16418518e-06, 1.26395317e-02],
                  [1.26395317e-02, 4.05925161e-02]],
                 [[3.48680871e-02, 1.26402319e-02],
                  [1.37904135e-02, 5.31904081e-07]]],
                [[[3.48680871e-02, 1.37904135e-02],
                  [1.26402319e-02, 5.31904081e-07]],
                 [[3.59539410e-07, -1.37902297e-02],
                  [-1.37902297e-02, -4.05933006e-02]]]])


@pytest.mark.skipif(world.size > 2,
                    reason='world.size > 2')
@pytest.mark.elph
def test_supercell(module_tmp_path, supercell_cache):
    # Generate supercell_cache
    supercell_cache

    # read supercell matrix
    g_xsNNMM, basis_info = Supercell.load_supercell_matrix()
    assert g_xsNNMM.shape == (6, 1, 2, 2, 2, 2)
    print(g_xsNNMM[0, 0])
    assert g_xsNNMM[0, 0] == pytest.approx(g00, rel=1e-2)
