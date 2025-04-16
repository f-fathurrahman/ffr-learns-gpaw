import pytest
from ase import Atoms

from my_gpaw import GPAW, LCAO
from my_gpaw.mixer import FFTMixer
from my_gpaw.test import equal


@pytest.mark.later
def test_lcao_fftmixer():
    bulk = Atoms('Li', pbc=True,
                 cell=[2.6, 2.6, 2.6])
    k = 4
    bulk.calc = GPAW(mode=LCAO(),
                     kpts=(k, k, k),
                     mixer=FFTMixer())
    e = bulk.get_potential_energy()
    equal(e, -1.710364, 1e-4)
