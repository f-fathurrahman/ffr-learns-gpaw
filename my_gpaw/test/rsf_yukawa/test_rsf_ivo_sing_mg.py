"""Test the calculation of the excitation energy of Na2 by RSF and IVOs."""
import pytest
from ase import Atoms
from ase.units import Hartree

import _gpaw
from my_gpaw import GPAW
from my_gpaw.cluster import Cluster
from my_gpaw.eigensolvers import RMMDIIS
from my_gpaw.lrtddft import LrTDDFT
from my_gpaw.mpi import world
from my_gpaw.occupations import FermiDirac
from my_gpaw.test import equal, gen


@pytest.mark.hybrids
def test_rsf_yukawa_rsf_ivo_sing_mg(in_tmp_dir, add_cwd_to_setup_paths):
    libxc_version = getattr(_gpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 3:
        from unittest import SkipTest
        raise SkipTest

    h = 0.35  # Gridspacing
    e_singlet = 4.61
    e_singlet_lr = 5.54

    gen('Mg', xcname='PBE', scalarrel=True, exx=True, yukawa_gamma=0.38)

    c = {'energy': 0.05, 'eigenstates': 3, 'density': 3}
    na2 = Cluster(Atoms('Mg', positions=[[0, 0, 0]]))
    na2.minimal_box(2.5, h=h)
    calc = GPAW(txt='mg_ivo.txt', xc='LCY-PBE:omega=0.38:excitation=singlet',
                eigensolver=RMMDIIS(), h=h, occupations=FermiDirac(width=0.0),
                spinpol=False, convergence=c)
    na2.calc = calc
    na2.get_potential_energy()
    (eps_homo, eps_lumo) = calc.get_homo_lumo()
    e_ex = eps_lumo - eps_homo
    equal(e_singlet, e_ex, 0.15)
    calc.write('mg.gpw')

    c2 = GPAW('mg.gpw')
    ihomo = int(c2.get_occupation_numbers().sum() / 2 + 0.5) - 1
    assert c2.hamiltonian.xc.excitation == 'singlet'
    lr = LrTDDFT(calc, txt='LCY_TDDFT_Mg.log',
                 restrict={'istart': ihomo, 'jend': ihomo + 1}, nspins=2)
    lr.write('LCY_TDDFT_Mg.ex.gz')
    if world.rank == 0:
        lr2 = LrTDDFT.read('LCY_TDDFT_Mg.ex.gz')
        lr2.diagonalize()
        ex_lr = lr2[1].get_energy() * Hartree
        equal(e_singlet_lr, ex_lr, 0.15)
