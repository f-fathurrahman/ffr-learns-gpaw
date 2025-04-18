
from math import sqrt

import numpy as np
from ase import Atoms

from my_gpaw import GPAW, restart
from my_gpaw.test import equal


def test_fileio_restart_density(in_tmp_dir):
    d = 3.0
    atoms = Atoms('Na3',
                  positions=[(0, 0, 0),
                             (0, 0, d),
                             (0, d * sqrt(3 / 4), d / 2)],
                  magmoms=[1.0, 1.0, 1.0],
                  cell=(3.5, 3.5, 4 + 2 / 3),
                  pbc=True)

    # Only a short, non-converged calculation
    conv = {'eigenstates': 1.e-3, 'energy': 1e-2, 'density': 1e-1}
    calc = GPAW(h=0.30, nbands=3,
                setups={'Na': '1'},
                convergence=conv)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    m0 = atoms.get_magnetic_moments()
    eig00 = calc.get_eigenvalues(spin=0)
    eig01 = calc.get_eigenvalues(spin=1)
    # Write the restart file
    calc.write('tmp.gpw')

    # Try restarting from all the files
    atoms, calc = restart('tmp.gpw')
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    m1 = atoms.get_magnetic_moments()
    eig10 = calc.get_eigenvalues(spin=0)
    eig11 = calc.get_eigenvalues(spin=1)
    print(e0, e1)
    equal(e0, e1, 2e-3)
    print(f0, f1)
    for ff0, ff1 in zip(f0, f1):
        err = np.linalg.norm(ff0 - ff1)
        # for forces we use larger tolerance
        equal(err, 0.0, 4e-2)
    print(m0, m1)
    for mm0, mm1 in zip(m0, m1):
        equal(mm0, mm1, 2e-3)
    print('A', eig00, eig10)
    for eig0, eig1 in zip(eig00, eig10):
        equal(eig0, eig1, 5e-3)
    print('B', eig01, eig11)
    for eig0, eig1 in zip(eig01, eig11):
        equal(eig0, eig1, 2e-2)

    # Check that after restart everythnig is writable
    calc.write('tmp2.gpw')
