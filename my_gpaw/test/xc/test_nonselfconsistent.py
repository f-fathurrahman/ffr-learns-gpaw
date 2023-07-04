from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW
from gpaw.test import equal


def xc(name):
    return {'name': name, 'stencil': 1}


def test_xc_nonselfconsistent(in_tmp_dir):
    a = 7.5 * Bohr
    n = 16
    atoms = Atoms('He', [(0.0, 0.0, 0.0)], cell=(a, a, a), pbc=True)
    params = dict(gpts=(n, n, n), nbands=1)
    atoms.calc = GPAW(**params, xc=xc('PBE'))
    e1 = atoms.get_potential_energy()
    e1ref = atoms.calc.get_reference_energy()
    de12 = atoms.calc.get_xc_difference(xc('revPBE'))
    atoms.calc = GPAW(**params, xc=xc('revPBE'))
    e2 = atoms.get_potential_energy()
    e2ref = atoms.calc.get_reference_energy()
    de21 = atoms.calc.get_xc_difference(xc('PBE'))
    print(e1ref + e1 + de12 - (e2ref + e2))
    print(e1ref + e1 - (e2ref + e2 + de21))
    print(de12, de21)
    equal(e1ref + e1 + de12, e2ref + e2, 8e-4)
    equal(e1ref + e1, e2ref + e2 + de21, 3e-3)

    atoms.calc.write('revPBE.gpw')

    de21b = GPAW('revPBE.gpw').get_xc_difference(xc('PBE'))
    equal(de21, de21b, 9e-8)

    energy_tolerance = 0.0005
    equal(e1, -0.07904951, energy_tolerance)
    equal(e2, -0.08147563, energy_tolerance)
