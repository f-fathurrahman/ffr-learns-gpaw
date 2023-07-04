import pytest
from gpaw import GPAW
from gpaw.utilities.ibz2bz import ibz2bz


@pytest.mark.legacy
@pytest.mark.serial
def test_ibz2bz(gpw_files, in_tmp_dir):
    """Test ibz.gpw -> bz.gpw utility."""
    path = 'bz.gpw'
    gpw = gpw_files['bcc_li_pw']
    ibz2bz(gpw, path)
    ef1 = GPAW(gpw).get_fermi_level()
    ef2 = GPAW(path).get_fermi_level()
    assert ef1 == ef2
