import pytest
from ase import Atoms

from my_gpaw25 import GPAW, LCAO
from my_gpaw25.mixer import FFTMixer


@pytest.mark.old_gpaw_only
def test_lcao_fftmixer():
    bulk = Atoms('Li', pbc=True,
                 cell=[2.6, 2.6, 2.6])
    k = 4
    bulk.calc = GPAW(mode=LCAO(),
                     kpts=(k, k, k),
                     mixer=FFTMixer())
    e = bulk.get_potential_energy()
    assert e == pytest.approx(-1.710364, abs=1e-4)
