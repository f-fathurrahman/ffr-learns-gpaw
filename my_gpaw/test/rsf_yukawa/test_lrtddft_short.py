"""Check TDDFT ionizations with Yukawa potential."""
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
from my_gpaw.test import equal


@pytest.mark.hybrids
def test_rsf_yukawa_lrtddft_short(in_tmp_dir):
    libxc_version = getattr(_gpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 3:
        from unittest import SkipTest
        raise SkipTest

    o_plus = Cluster(Atoms('Be', positions=[[0, 0, 0]]))
    o_plus.set_initial_magnetic_moments([1.0])
    o_plus.minimal_box(2.5, h=0.35)

    def get_paw(**kwargs):
        """Return calculator object."""
        c = {'energy': 0.05, 'eigenstates': 0.05, 'density': 0.05}
        return GPAW(convergence=c, eigensolver=RMMDIIS(),
                    nbands=3,
                    xc='PBE',
                    parallel={'domain': world.size}, h=0.35,
                    occupations=FermiDirac(width=0.0, fixmagmom=True),
                    **kwargs)

    calc_plus = get_paw(txt='Be_plus_PBE.log', charge=1)
    o_plus.calc = calc_plus
    o_plus.get_potential_energy()

    calc_plus = calc_plus.new(xc='LCY-PBE:omega=0.83:unocc=True',
                              experimental={'niter_fixdensity': 2},
                              txt='Be_plus_LCY_PBE_083.log')
    o_plus.calc = calc_plus
    o_plus.get_potential_energy()

    lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_Be.log',
                 restrict={'istart': 0, 'jend': 1})
    equal(lr.xc.omega, 0.83)
    lr.write('LCY_TDDFT_Be.ex.gz')
    e_ion = 9.3
    ip_i = 13.36
    # reading is problematic with EXX on more than one core
    if world.rank == 0:
        lr2 = LrTDDFT.read('LCY_TDDFT_Be.ex.gz')
        lr2.diagonalize()
        equal(lr2.xc.omega, 0.83)
        ion_i = lr2[0].get_energy() * Hartree + e_ion
        equal(ion_i, ip_i, 0.3)
