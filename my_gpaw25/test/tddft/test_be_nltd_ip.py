import numpy as np
from ase import Atoms
from my_gpaw25.tddft import TDDFT
from my_gpaw25.tddft.abc import LinearAbsorbingBoundary
from my_gpaw25.tddft.laser import CWField
import pytest


@pytest.mark.old_gpaw_only
def test_tddft_be_nltd_ip(in_tmp_dir, gpw_files):
    atoms = Atoms('Be', [(0, 0, 0)], pbc=False)
    atoms.center(vacuum=6)

    td_calc = TDDFT(gpw_files['be_atom_fd'],
                    td_potential=CWField(1e-3, 2.0 * np.pi / 50.0, 150.0))
    td_calc.set_absorbing_boundary(
        LinearAbsorbingBoundary(5.0, 0.01,
                                atoms.positions.copy()))
    td_calc.propagate(8.0, 5)
    td_calc.write('be_nl_td.gpw', 'all')

    td_rest = TDDFT('be_nl_td.gpw',
                    td_potential=CWField(1e-3, 2.0 * np.pi / 50.0, 150.0))
    td_rest.set_absorbing_boundary(
        LinearAbsorbingBoundary(5.0, 0.01,
                                atoms.positions.copy()))
    td_rest.propagate(8.0, 5)
    td_calc.write('be_nl_td.gpw', 'all')
