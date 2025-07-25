from ase import Atoms
from my_gpaw25 import GPAW


def test_ae_potential():
    h = Atoms('H',
              cell=[2.01, 2.01, 2.01],
              pbc=True)
    h.calc = GPAW(
        mode='lcao',
        setups='ae',
        xc='PBE')
    f = h.get_forces()
    assert abs(f).max() < 1e-14
