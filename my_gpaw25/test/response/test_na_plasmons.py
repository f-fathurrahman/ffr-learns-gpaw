import pytest
from my_gpaw25.mpi import world
from ase import Atoms
from my_gpaw25 import GPAW, PW
from my_gpaw25.response.df import DielectricFunction
from my_gpaw25.test import findpeak

# Comparing the EELS spectrum of sodium for different block
# parallelizations. Intended to be run with 8 cores.
# Reasons that this can fail:
# - Bug in block parallelization

pytestmark = pytest.mark.skipif(world.size < 4, reason='world.size < 4')


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.slow
def test_response_na_plasmons(in_tmp_dir, scalapack):
    a = 4.23 / 2.0
    a1 = Atoms('Na',
               scaled_positions=[[0, 0, 0]],
               cell=(a, a, a),
               pbc=True)

    a1.calc = GPAW(mode=PW(300),
                   kpts={'size': (10, 10, 10), 'gamma': True},
                   parallel={'band': 1},
                   txt='small.txt')

    a1.get_potential_energy()
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a1.calc.write('gs_Na.gpw', 'all')

    # Calculate the dielectric functions
    df1 = DielectricFunction('gs_Na.gpw',
                             nblocks=1,
                             ecut=40,
                             rate=0.001,
                             txt='1block.txt')

    df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')

    df2 = DielectricFunction('gs_Na.gpw',
                             nblocks=4,
                             ecut=40,
                             rate=0.001,
                             txt='4block.txt')

    df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')

    # Compare plasmon frequencies and intensities
    w_w = df1.chi0calc.wd.omega_w
    w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
    w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
    assert w1 == pytest.approx(w2, abs=1e-2)
    assert I1 == pytest.approx(I2, abs=1e-3)
