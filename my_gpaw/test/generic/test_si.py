from gpaw import GPAW, FermiDirac
from ase import Atoms
from gpaw.test import equal
import numpy as np


def test_generic_si(in_tmp_dir):
    a = 5.404
    bulk = Atoms(symbols='Si8',
                 scaled_positions=[(0, 0, 0),
                                   (0, 0.5, 0.5),
                                   (0.5, 0, 0.5),
                                   (0.5, 0.5, 0),
                                   (0.25, 0.25, 0.25),
                                   (0.25, 0.75, 0.75),
                                   (0.75, 0.25, 0.75),
                                   (0.75, 0.75, 0.25)],
                 pbc=True, cell=(a, a, a))
    n = 20
    calc = GPAW(gpts=(n, n, n),
                nbands=8 * 3,
                occupations=FermiDirac(width=0.01),
                kpts=(1, 1, 1))
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    eigs = calc.get_eigenvalues(kpt=0)
    calc.write('temp.gpw')

    calc = GPAW('temp.gpw').fixed_density()
    eigs2 = calc.get_eigenvalues(kpt=0)
    print('Orginal', eigs)
    print('Fixdensity', eigs2)
    print('Difference', eigs2 - eigs)

    assert np.fabs(eigs2 - eigs)[:-1].max() < 3e-5
    equal(e1, -36.767, 0.003)
