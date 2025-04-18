# listserv.fysik.dtu.dk/pipermail/gpaw-developers/2014-February/004374.html
from my_gpaw import GPAW
from my_gpaw.test import equal
from ase.build import molecule


def test_pathological_LDA_unstable():
    for i in range(12):
        mol = molecule('H2')
        mol.center(vacuum=1.5)
        calc = GPAW(h=0.3, nbands=2, mode='lcao', txt=None, basis='sz(dzp)',
                    xc='oldLDA',
                    convergence={'maximum iterations': 1})
        mol.calc = calc
        e = mol.get_potential_energy()
        if i == 0:
            eref = e
        if calc.wfs.world.rank == 0:
            print(repr(e))
        equal(e - eref, 0, 1.e-12)
