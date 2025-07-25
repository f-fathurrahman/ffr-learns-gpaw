import pytest
from my_gpaw25 import GPAW
from my_gpaw25.xc.rpa import RPACorrelation
from my_gpaw25.mpi import serial_comm


@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_Na(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['na_pw'], communicator=serial_comm)
    ecut = 120
    rpa = RPACorrelation(calc, txt=f'rpa_{ecut}s.txt', ecut=[ecut])
    E = rpa.calculate()
    assert E == pytest.approx(-1.106, abs=0.005)
