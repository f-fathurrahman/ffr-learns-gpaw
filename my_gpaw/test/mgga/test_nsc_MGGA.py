import pytest
from ase import Atoms
from my_gpaw import GPAW, Mixer, Davidson
from my_gpaw.test import equal

# ??? g = Generator('H', 'TPSS', scalarrel=True, nofiles=True)


@pytest.mark.mgga
def test_mgga_nsc_MGGA(in_tmp_dir):
    def xc(name):
        return dict(name=name, stencil=1)

    atoms = Atoms('H', magmoms=[1], pbc=True)
    atoms.center(vacuum=3)

    def getkwargs():
        return dict(eigensolver=Davidson(3),
                    mixer=Mixer(0.7, 5, 50.0),
                    parallel=dict(augment_grids=True),
                    gpts=(32, 32, 32), nbands=1, xc=xc('oldPBE'))

    calc = GPAW(txt='Hnsc.txt', **getkwargs())
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    _ = calc.get_reference_energy()
    de12t = calc.get_xc_difference(xc('TPSS'))
    de12m = calc.get_xc_difference(xc('M06-L'))
    de12r = calc.get_xc_difference(xc('revTPSS'))

    print('================')
    print('e1 = ', e1)
    print('de12t = ', de12t)
    print('de12m = ', de12m)
    print('de12r = ', de12r)
    print('tpss = ', e1 + de12t)
    print('m06l = ', e1 + de12m)
    print('revtpss = ', e1 + de12r)
    print('================')

    equal(e1 + de12t, -1.11723235592, 0.005)
    equal(e1 + de12m, -1.18207312133, 0.005)
    equal(e1 + de12r, -1.10093196353, 0.005)

    # ??? g = Generator('He', 'TPSS', scalarrel=True, nofiles=True)

    atomsHe = Atoms('He', pbc=True)
    atomsHe.center(vacuum=3)
    calc = GPAW(txt='Hensc.txt', **getkwargs())
    atomsHe.calc = calc
    e1He = atomsHe.get_potential_energy()
    _ = calc.get_reference_energy()
    de12tHe = calc.get_xc_difference(xc('TPSS'))
    de12mHe = calc.get_xc_difference(xc('M06-L'))
    de12rHe = calc.get_xc_difference(xc('revTPSS'))

    print('================')
    print('e1He = ', e1He)
    print('de12tHe = ', de12tHe)
    print('de12mHe = ', de12mHe)
    print('de12rHe = ', de12rHe)
    print('tpss = ', e1He + de12tHe)
    print('m06l = ', e1He + de12mHe)
    print('revtpss = ', e1He + de12rHe)
    print('================')

    equal(e1He + de12tHe, -0.409972893501, 0.005)
    equal(e1He + de12mHe, -0.487249688866, 0.005)
    equal(e1He + de12rHe, -0.447232286813, 0.005)

    energy_tolerance = 0.001
    equal(e1, -1.124, energy_tolerance)
    equal(e1He, 0.0100192, energy_tolerance)
