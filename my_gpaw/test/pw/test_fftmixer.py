from ase import Atoms
from my_gpaw import GPAW
from my_gpaw.mixer import FFTMixer
from my_gpaw import PW
from my_gpaw.test import equal


def test_pw_fftmixer():
    bulk = Atoms('Li', pbc=True,
                 cell=[2.6, 2.6, 2.6])
    k = 4
    bulk.calc = GPAW(mode=PW(200),
                     kpts=(k, k, k),
                     mixer=FFTMixer(),
                     eigensolver='rmm-diis')
    e = bulk.get_potential_energy()
    equal(e, -1.98481281259, 1.0e-4)
