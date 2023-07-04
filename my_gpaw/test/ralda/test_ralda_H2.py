import pytest

from gpaw.mpi import world
from gpaw.xc.fxc import FXCCorrelation


@pytest.mark.rpa
@pytest.mark.response
@pytest.mark.parametrize('gpw, kwargs, ref_energy, abstol', [
    ('h2_pw210_rmmdiis_wfs', dict(xc='rALDA', nblocks=min(4, world.size)),
     -0.8411, 1e-3),
    ('h2_pw210_rmmdiis_wfs', dict(xc='rAPBE'), -0.7389, 1e-3),
    ('h_pw210_rmmdiis_wfs', dict(xc='rALDA'), 0.0029, 1e-4),
    ('h_pw210_rmmdiis_wfs', dict(xc='rAPBE', nblocks=min(4, world.size)),
     0.0133, 1e-4),
])
def test_ralda_energy_H2(in_tmp_dir, gpw_files, scalapack, gpw,
                         kwargs,
                         ref_energy, abstol):
    gpw = gpw_files[gpw]
    fxc = FXCCorrelation(gpw, **kwargs, ecut=[200])

    energy = fxc.calculate()
    assert energy == pytest.approx(ref_energy, abs=abstol)
