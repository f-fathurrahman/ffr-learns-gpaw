import pytest
from my_gpaw.mpi import world
import numpy as np
from ase.build import bulk
from my_gpaw import GPAW
from my_gpaw.response.df import DielectricFunction
from my_gpaw.response.bse import BSE, read_spectrum
from my_gpaw.test import findpeak, equal

pytestmark = pytest.mark.skipif(world.size < 4,
                                reason='world.size < 4')


@pytest.mark.response
def test_response_bse_aluminum(in_tmp_dir):
    GS = 1
    df = 1
    bse = 1
    check_spectrum = 1

    if GS:
        a = 4.043
        atoms = bulk('Al', 'fcc', a=a)
        calc = GPAW(mode='pw',
                    kpts={'size': (4, 4, 4), 'gamma': True},
                    xc='LDA',
                    nbands=4,
                    convergence={'bands': 'all'})

        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('Al.gpw', 'all')

    q_c = np.array([0.25, 0.0, 0.0])
    w_w = np.linspace(0, 24, 241)
    eta = 0.2
    ecut = 50
    if bse:
        bse = BSE('Al.gpw',
                  valence_bands=range(4),
                  conduction_bands=range(4),
                  mode='RPA',
                  nbands=4,
                  ecut=ecut,
                  write_h=False,
                  write_v=False,
                  )
        bse.get_eels_spectrum(filename='bse_eels.csv',
                              q_c=q_c,
                              w_w=w_w,
                              eta=eta)
        omega_w, bse_w = read_spectrum('bse_eels.csv')

    if df:
        df = DielectricFunction(calc='Al.gpw',
                                frequencies=w_w,
                                eta=eta,
                                ecut=ecut,
                                hilbert=False)
        df_w = df.get_eels_spectrum(q_c=q_c, filename=None)[1]

    if check_spectrum:
        assert w_w == pytest.approx(omega_w)
        w_ = 15.1423
        I_ = 25.4359
        wbse, Ibse = findpeak(w_w, bse_w)
        wdf, Idf = findpeak(w_w, df_w)
        equal(wbse, w_, 0.01)
        equal(wdf, w_, 0.01)
        equal(Ibse, I_, 0.1)
        equal(Idf, I_, 0.1)
