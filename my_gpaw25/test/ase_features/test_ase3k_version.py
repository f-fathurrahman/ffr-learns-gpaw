from packaging.version import Version
from my_gpaw25 import __ase_version_required__
from ase import __version__


def test_ase_features_ase3k_version():
    assert Version(__version__) >= Version(__ase_version_required__)
