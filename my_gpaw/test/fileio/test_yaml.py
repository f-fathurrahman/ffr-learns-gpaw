import pytest
from ase.io import read


@pytest.mark.skip
def test_yaml(gpw_files):
    pytest.importorskip('yaml')
    print(gpw_files['h2_pw'].with_name('h2_pw.txt'))
    a = read(gpw_files['h2_pw'].with_name('h2_pw.txt'))
    print(a)
