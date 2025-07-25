import numpy as np
import pytest
from ase import Atoms

from my_gpaw25 import GPAW
from my_gpaw25.mom import prepare_mom_calculation


@pytest.mark.mom
def test_mom_lcao_forces(in_tmp_dir):
    f_sn = [[1., 1., 1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 1., 0., 0.]]
    L = 4.0
    d = 1.13
    delta = 0.01

    atoms = Atoms('CO',
                  [[0.5 * L, 0.5 * L, 0.5 * L - 0.5 * d],
                   [0.5 * L, 0.5 * L, 0.5 * L + 0.5 * d]])
    atoms.set_cell([L, L, L])
    atoms.rotate(1, 'x', center=[0.5 * L, 0.5 * L, 0.5 * L])

    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=7,
                h=0.24,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-4})

    atoms.calc = calc
    occ = prepare_mom_calculation(calc, atoms, f_sn)
    F = atoms.get_forces()

    # Test overlaps
    occ.initialize_reference_orbitals()
    for kpt in calc.wfs.kpt_u:
        f_n = calc.get_occupation_numbers(spin=kpt.s)
        unoccupied = [True for i in range(len(f_n))]
        P = occ.calculate_weights(kpt, 1.0, unoccupied)
        assert (np.allclose(P, f_n))

    E = []
    p = atoms.positions.copy()
    for i in [-1, 1]:
        pnew = p.copy()
        pnew[0, 2] -= delta / 2. * i
        pnew[1, 2] += delta / 2. * i
        atoms.set_positions(pnew)

        E.append(atoms.get_potential_energy())

    f = np.sqrt(((F[1, :] - F[0, :])**2).sum()) * 0.5
    fnum = (E[0] - E[1]) / (2. * delta)  # central difference

    print(fnum, f)
    assert fnum == pytest.approx(11.52, abs=0.016)
    assert f == pytest.approx(fnum, abs=0.1)
