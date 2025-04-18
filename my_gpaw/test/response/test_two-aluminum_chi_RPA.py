import pytest
import numpy as np
import time

from ase.build import bulk
from ase.parallel import parprint

from my_gpaw import GPAW, PW
from my_gpaw.test import findpeak, equal
from my_gpaw.mpi import size, world

from my_gpaw.response import ResponseGroundStateAdapter
from my_gpaw.response.chiks import ChiKSCalculator
from my_gpaw.response.susceptibility import ChiFactory
from my_gpaw.response.pair_functions import read_susceptibility_array


@pytest.mark.kspair
@pytest.mark.response
def test_response_two_aluminum_chi_RPA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 4.043
    atoms1 = bulk('Al', 'fcc', a=a)
    atoms2 = atoms1.repeat((2, 1, 1))

    calc1 = GPAW(mode=PW(200),
                 nbands=4,
                 kpts=(8, 8, 8),
                 parallel={'domain': 1},
                 xc='LDA')

    atoms1.calc = calc1
    atoms1.get_potential_energy()

    t2 = time.time()

    calc2 = GPAW(mode=PW(200),
                 nbands=8,
                 kpts=(4, 8, 8),
                 parallel={'domain': 1},
                 xc='LDA')

    atoms2.calc = calc2
    atoms2.get_potential_energy()

    t3 = time.time()

    # Excited state calculation
    q1_qc = [np.array([1 / 8., 0., 0.]), np.array([3 / 8., 0., 0.])]
    q2_qc = [np.array([1 / 4., 0., 0.]), np.array([- 1 / 4., 0., 0.])]
    w = np.linspace(0, 24, 241)

    # Calculate susceptibility using Al1
    calculate_chi(calc1, q1_qc, w, filename_prefix='Al1')

    t4 = time.time()

    # Calculate susceptibility using Al2
    calculate_chi(calc2, q2_qc, w, filename_prefix='Al2')

    t5 = time.time()

    parprint('')
    parprint('Ground state calc 1 took', (t2 - t1), 'seconds')
    parprint('Ground state calc 2 took', (t3 - t2), 'seconds')
    parprint('Susceptibility calc 1 took', (t4 - t3), 'seconds')
    parprint('Susceptibility calc 2 took', (t5 - t4), 'seconds')

    # Check that results are consistent, when structure is simply repeated

    # Read results
    w11_w, G11_Gc, chi11_wGG = read_susceptibility_array('Al1_chiGG_q1.pckl')
    w21_w, G21_Gc, chi21_wGG = read_susceptibility_array('Al2_chiGG_q1.pckl')
    w12_w, G12_Gc, chi12_wGG = read_susceptibility_array('Al1_chiGG_q2.pckl')
    w22_w, G22_Gc, chi22_wGG = read_susceptibility_array('Al2_chiGG_q2.pckl')

    # Check that reciprocal lattice vectors remain as assumed in check below
    equal(np.linalg.norm(G11_Gc[0]), 0., 1e-6)
    equal(np.linalg.norm(G21_Gc[0]), 0., 1e-6)
    equal(np.linalg.norm(G12_Gc[0]), 0., 1e-6)
    equal(np.linalg.norm(G22_Gc[1] - np.array([1, 0, 0])), 0., 1e-6)

    # Check plasmon peaks remain the same
    wpeak11, Ipeak11 = findpeak(w11_w, -chi11_wGG[:, 0, 0].imag)
    wpeak21, Ipeak21 = findpeak(w21_w, -chi21_wGG[:, 0, 0].imag)
    equal(wpeak11, wpeak21, 0.02)
    equal(Ipeak11, Ipeak21, 1.0)
    wpeak12, Ipeak12 = findpeak(w12_w, -chi12_wGG[:, 0, 0].imag)
    wpeak22, Ipeak22 = findpeak(w22_w, -chi22_wGG[:, 1, 1].imag)
    equal(wpeak12, wpeak22, 0.05)
    equal(Ipeak12, Ipeak22, 1.0)


def calculate_chi(calc, q_qc, w,
                  eta=0.2, ecut=50,
                  spincomponent='00', fxc='RPA',
                  filename_prefix=None, reduced_ecut=25):
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs, ecut=ecut)
    chi_factory = ChiFactory(chiks_calc)

    if filename_prefix is None:
        filename = 'chiGG_qXXX.pckl'
    else:
        filename = filename_prefix + '_chiGG_qXXX.pckl'

    for q, q_c in enumerate(q_qc):
        fname = filename.replace('XXX', str(q + 1))
        _, chi = chi_factory(spincomponent, q_c, w + 1.j * eta, fxc=fxc)
        chi = chi.copy_with_reduced_ecut(reduced_ecut)
        chi.write_array(fname)
        world.barrier()
