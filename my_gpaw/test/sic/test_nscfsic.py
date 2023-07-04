import pytest
from gpaw.mpi import world
from ase import Atoms
from gpaw import GPAW
from gpaw.utilities.sic import NSCFSIC

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.later
def test_sic_nscfsic(in_tmp_dir):
    atoms = ['He', 'Be']  # ,'Ne'] # Ne deviates already 2.5 eV
    EE = []
    EREF = [-79.4, -399.8, -3517.6]

    for a in atoms:
        s = Atoms(a)
        s.center(vacuum=4.0)
        calc = GPAW(h=0.15, txt=a + '.txt')
        s.calc = calc
        s.get_potential_energy()
        EE.append(NSCFSIC(calc).calculate())

    print("Difference to table VI of Phys. Rev. B 23, 5048 in eV")
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.23.5048
    print("%10s%10s%10s%10s" % ("atom", "ref.", "gpaw", "diff"))
    for a, er, e in zip(atoms, EREF, EE):
        print("%10s%10.2f%10.2f%10.2f" % (a, er, e, er - e))
        assert abs(er - e) < 0.1
        # Arbitrary 0.1 eV tolerance for non-self consistent SIC
        # Note that Ne already deviates 2.5 eV
