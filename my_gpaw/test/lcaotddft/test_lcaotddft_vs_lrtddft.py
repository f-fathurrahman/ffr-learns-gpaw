from pathlib import Path

import pytest
import numpy as np

from ase.build import molecule
from ase.units import Hartree
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum as spec_td
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft import photoabsorption_spectrum as spec_lr
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.mpi import world, serial_comm, broadcast

from gpaw.test import only_on_master


pytestmark = [pytest.mark.usefixtures('module_tmp_path')]


@pytest.fixture(scope='module')
@only_on_master(world, broadcast)
def ground_state_calculation():
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)
    calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                basis='sz(dzp)', mode='lcao', xc='oldLDA',
                convergence={'density': 1e-8},
                communicator=serial_comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()

    gpw_fpath = Path('gs.gpw').resolve()
    calc.write(gpw_fpath, mode='all')
    return gpw_fpath


@pytest.fixture(scope='module')
def time_propagation_calculation(ground_state_calculation):
    td_calc = LCAOTDDFT(ground_state_calculation, txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0, 0, 1e-5])
    td_calc.propagate(30, 150)
    spec_td('dm.dat', 'spec_td.dat',
            e_min=0, e_max=10, width=0.5, delta_e=0.1)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_td.dat')
    spec_e = data_ej[:, 3]
    spec_e[1:] /= data_ej[1:, 0]
    return spec_e


@pytest.fixture(scope='module')
def lrtddft_calculation(ground_state_calculation):
    calc = GPAW(ground_state_calculation, txt=None)
    lr = LrTDDFT(calc, xc='LDA', txt='lr.out')
    lr.diagonalize()
    spec_lr(lr, 'spec_lr.dat',
            e_min=0, e_max=10, width=0.5, delta_e=0.1)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_lr.dat')
    spec_e = data_ej[:, 4]
    spec_e[1:] /= lr[0].get_energy() * Hartree
    return spec_e


@pytest.fixture(scope='module')
def lrtddft2_calculation(ground_state_calculation):
    calc = GPAW(ground_state_calculation, txt='lr2.out')
    lr2 = LrTDDFT2('lr2', calc, fxc='LDA')
    lr2.calculate()
    lr2.get_spectrum('spec_lr2.dat', 0, 10.1, 0.1, width=0.5)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_lr2.dat')
    spec_e = data_ej[:, 5]
    spec_e[1:] /= lr2.lr_transitions.get_transitions()[0][0]
    return spec_e


@pytest.mark.rttddft
def test_lcaotddft_vs_lrtddft(time_propagation_calculation,
                              lrtddft_calculation):
    # One can decrease the tolerance by decreasing the time step
    # and other parameters
    assert (time_propagation_calculation
            == pytest.approx(lrtddft_calculation, abs=1e-2))


@pytest.mark.rttddft
def test_lcaotddft_vs_lrtddft2(time_propagation_calculation,
                               lrtddft2_calculation):
    # One can decrease the tolerance by decreasing the time step
    # and other parameters
    assert (time_propagation_calculation
            == pytest.approx(lrtddft2_calculation, abs=1e-2))


@pytest.mark.rttddft
def test_lrtddft_vs_lrtddft2(lrtddft_calculation,
                             lrtddft2_calculation):
    assert (lrtddft_calculation
            == pytest.approx(lrtddft2_calculation, abs=1e-3))
