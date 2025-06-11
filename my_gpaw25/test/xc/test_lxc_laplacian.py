"""check if an error is raised if the laplacian is needed (mgga)."""
import pytest
from my_gpaw25.xc import LibXC
from my_gpaw25.xc.libxc import FunctionalNeedsLaplacianError


@pytest.mark.mgga
@pytest.mark.libxc
def test_mgga_lxc_laplacian():
    """Check for raised error."""
    with pytest.raises(FunctionalNeedsLaplacianError):
        LibXC('MGGA_X_BR89+MGGA_C_TPSS', disable_fhc=False)


def test_mgga_lxc_suppressed_laplacian():
    """Check for suppressed error."""
    LibXC('MGGA_X_BR89+MGGA_C_TPSS', provides_laplacian=True)
