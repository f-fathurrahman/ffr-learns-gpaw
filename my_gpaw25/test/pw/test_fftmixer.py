from ase import Atoms
from my_gpaw25 import GPAW
from my_gpaw25.mixer import FFTMixer
from my_gpaw25 import PW
import pytest


def test_pw_fftmixer():
    bulk = Atoms('Li', pbc=True,
                 cell=[2.6, 2.6, 2.6])
    k = 4
    bulk.calc = GPAW(mode=PW(200),
                     kpts=(k, k, k),
                     mixer=FFTMixer(),
                     eigensolver='rmm-diis')
    e = bulk.get_potential_energy()
    assert e == pytest.approx(-1.98481281259, abs=1.0e-4)
