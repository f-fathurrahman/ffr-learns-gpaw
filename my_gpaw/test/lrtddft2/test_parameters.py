import pytest

from contextlib import contextmanager
from pathlib import Path

from ase.build import molecule
from my_gpaw import GPAW
from my_gpaw.lrtddft2 import LrTDDFT2


@contextmanager
def no_error():
    yield


pytestmark = pytest.mark.usefixtures('module_tmp_path')


@pytest.fixture(scope='module')
def ground_state():
    atoms = molecule('H2O')
    atoms.center(vacuum=4)

    calc = GPAW(h=0.4, mode='lcao', basis='sz(dzp)', txt='gs.out',
                nbands=6, xc='LDA')
    atoms.calc = calc
    atoms.get_potential_energy()
    gpw_fpath = Path('gs.gpw').resolve()
    calc.write(gpw_fpath, mode='all')
    return gpw_fpath


@pytest.mark.lrtddft
@pytest.mark.parametrize('max_energy_diff, expectation', [
    (15, pytest.raises(RuntimeError, match=r'.* HOMO \(n=3\) .*')),
    (10, no_error())])
def test_max_energy_diff(ground_state, max_energy_diff, expectation,
                         in_tmp_dir):
    calc = GPAW(ground_state)
    with expectation:
        LrTDDFT2('lr2', calc, fxc='LDA', max_energy_diff=max_energy_diff)


@pytest.mark.lrtddft
@pytest.mark.parametrize('min_occ', [None, 0])
@pytest.mark.parametrize('min_unocc', [None, 0])
@pytest.mark.parametrize('max_occ', [None, 5])
@pytest.mark.parametrize('max_unocc', [None, 5])
def test_indices_with_max_energy_diff(ground_state, min_occ, min_unocc,
                                      max_occ, max_unocc, in_tmp_dir):
    calc = GPAW(ground_state)

    if (min_occ is None or min_unocc is None
        or max_occ is None or max_unocc is None):
        expectation = pytest.raises(RuntimeError)
    else:
        expectation = no_error()

    with expectation:
        LrTDDFT2('lr2', calc, fxc='LDA', max_energy_diff=15,
                 min_occ=min_occ, min_unocc=min_unocc,
                 max_occ=max_occ, max_unocc=max_unocc)


@pytest.mark.lrtddft
@pytest.mark.parametrize('min_occ', [None, 0])
@pytest.mark.parametrize('min_unocc', [None, 0])
@pytest.mark.parametrize('max_occ', [None, 5])
@pytest.mark.parametrize('max_unocc', [None, 5])
def test_indices(ground_state, min_occ, min_unocc,
                 max_occ, max_unocc, in_tmp_dir):
    calc = GPAW(ground_state)
    LrTDDFT2('lr2', calc, fxc='LDA',
             min_occ=min_occ, min_unocc=min_unocc,
             max_occ=max_occ, max_unocc=max_unocc)
