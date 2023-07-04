# this test should coverage the save and restore of
# fermi-levels when using fixmagmom:
#
# yes, fermi-level-splitting sounds a little bit strange

from ase import Atoms
from my_gpaw import GPAW, FermiDirac, MixerSum
from my_gpaw.test import equal


def test_fermisplit(in_tmp_dir):
    calc = GPAW(occupations=FermiDirac(width=0.1, fixmagmom=True),
                mixer=MixerSum(beta=0.05, nmaxold=3, weight=50.0),
                convergence={'energy': 0.1, 'eigenstates': 1.5e-1,
                             'density': 1.5e-1})
    atoms = Atoms('Cr', pbc=False)
    atoms.center(vacuum=4)
    mm = [1] * 1
    mm[0] = 6.
    atoms.set_initial_magnetic_moments(mm)
    atoms.calc = calc
    atoms.get_potential_energy()

    ef1 = calc.get_fermi_levels().mean()
    efsplit1 = calc.get_fermi_levels().ptp()

    ef3 = calc.get_fermi_levels()
    calc.write('test.gpw')

    # check number one: is the splitting value saved?
    readtest = GPAW('test.gpw')
    ef2 = readtest.get_fermi_levels().mean()
    efsplit2 = readtest.get_fermi_levels().ptp()

    # numpy arrays
    ef4 = readtest.get_fermi_levels()

    # These values should be identic
    equal(ef1, ef2, 1e-9)
    equal(efsplit1, efsplit2, 1e-9)
    equal(ef3.mean(), ef1, 1e-9)
    equal(ef3.mean(), ef2, 1e-9)
    equal(ef3.mean(), ef4.mean(), 1e-9)
    equal(ef3[0] - ef3[1], ef4[0] - ef4[1], 1e-9)
    equal(efsplit1, ef4[0] - ef4[1], 1e-9)
