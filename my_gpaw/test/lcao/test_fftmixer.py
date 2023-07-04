import pytest
from ase import Atoms

from gpaw import GPAW, LCAO
from gpaw.mixer import FFTMixer
from gpaw.test import equal


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
