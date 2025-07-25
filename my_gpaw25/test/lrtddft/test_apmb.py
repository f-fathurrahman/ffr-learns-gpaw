import pytest
from ase import Atom, Atoms
from ase.parallel import parprint

from my_gpaw25 import GPAW, mpi
from my_gpaw25.lrtddft import LrTDDFT


@pytest.mark.lrtddft
@pytest.mark.libxc
def test_lrtddft_apmb():
    txt = '-'
    txt = '/dev/null'

    load = False

    if not load:
        R = 0.7  # approx. experimental bond length
        a = 3.0
        c = 4.0
        H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                    Atom('H', (a / 2, a / 2, (c + R) / 2))],
                   cell=(a, a, c))
        calc = GPAW(mode='fd', xc='PBE', nbands=2, spinpol=False, txt=txt)
        H2.calc = calc
        H2.get_potential_energy()
    else:
        calc = GPAW('H2.gpw', txt=txt)

    # DFT only

    xc = 'LDA'

    # no spin

    lr = LrTDDFT(calc, xc=xc)
    lr.diagonalize()

    lr_ApmB = LrTDDFT(calc, xc=xc, force_ApmB=True)
    lr_ApmB.diagonalize()
    parprint('lr=', lr)
    parprint('ApmB=', lr_ApmB)
    assert lr[0].get_energy() == pytest.approx(lr_ApmB[0].get_energy(),
                                               abs=5.e-9)

    # with spin
    parprint('------ with spin')

    if not load:
        c_spin = GPAW(mode='fd', xc='PBE', nbands=2,
                      spinpol=True, parallel={'domain': mpi.world.size},
                      txt=txt)
        H2.calc = c_spin
        c_spin.calculate(H2)
    else:
        c_spin = GPAW('H2spin.gpw', txt=txt)
    lr = LrTDDFT(c_spin, xc=xc)
    lr.diagonalize()

    lr_ApmB = LrTDDFT(c_spin, xc=xc, force_ApmB=True)
    lr_ApmB.diagonalize()
    parprint('lr=', lr)
    parprint('ApmB=', lr_ApmB)
    assert lr[0].get_energy() == pytest.approx(lr_ApmB[0].get_energy(),
                                               abs=5.e-8)
    assert lr[1].get_energy() == pytest.approx(lr_ApmB[1].get_energy(),
                                               abs=5.e-8)

    # with spin virtual
    parprint('------ with virtual spin')

    lr = LrTDDFT(calc, xc=xc, nspins=2)
    lr.diagonalize()

    # ApmB
    lr_ApmB = LrTDDFT(calc, xc=xc, nspins=2)
    lr_ApmB.diagonalize()
    parprint('lr=', lr)
    parprint('ApmB=', lr_ApmB)
    assert lr[0].get_energy() == pytest.approx(lr_ApmB[0].get_energy(),
                                               abs=5.e-8)
    assert lr[1].get_energy() == pytest.approx(lr_ApmB[1].get_energy(),
                                               abs=5.e-8)

    # with HF exchange

    xc = 'PBE0'

    parprint('------ with spin xc=', xc)
    lr_spin = LrTDDFT(c_spin, xc=xc)
    lr_spin.diagonalize()
    parprint('lr=', lr_spin)

    parprint('------ with virtual spin xc=', xc)
    lr = LrTDDFT(calc, xc=xc, nspins=2)
    lr.diagonalize()
    parprint('lr=', lr)
    assert lr[0].get_energy() == pytest.approx(lr_spin[0].get_energy(),
                                               abs=3.8e-6)
    assert lr[1].get_energy() == pytest.approx(lr_spin[1].get_energy(),
                                               abs=3.4e-6)
