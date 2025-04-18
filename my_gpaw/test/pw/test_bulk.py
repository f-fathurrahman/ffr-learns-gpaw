import numpy as np
from ase import Atoms
from my_gpaw import GPAW
from my_gpaw import PW
from my_gpaw.test import equal


def test_pw_bulk():
    bulk = Atoms('Li', pbc=True)
    k = 4
    calc = GPAW(mode=PW(200),
                kpts=(k, k, k),
                eigensolver='rmm-diis')

    bulk.calc = calc
    e = []
    A = [2.6, 2.65, 2.7, 2.75, 2.8]
    for a in A:
        bulk.set_cell((a, a, a))
        e.append(bulk.get_potential_energy())

    a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
    print('a =', a)
    equal(a, 2.65247379609, 0.001)
