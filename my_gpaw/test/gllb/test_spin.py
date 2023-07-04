import pytest
from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.test import equal


@pytest.mark.gllb
@pytest.mark.libxc
def test_gllb_spin():
    for spin in [False, True]:
        a = 3.56
        atoms = bulk('C', 'diamond', a=a)
        calc = GPAW(kpts=(3, 3, 3),
                    xc='GLLBSC',
                    spinpol=spin,
                    nbands=8,
                    convergence={'bands': 6, 'density': 1e-6},
                    occupations=FermiDirac(width=0.005))
        atoms.calc = calc
        atoms.get_potential_energy()
        resp = calc.hamiltonian.xc.response
        # Eks is the Kohn-Sham gap and Dxc is the derivative discontinuity
        if spin:
            homoa, lumoa = calc.get_homo_lumo(spin=0)
            homob, lumob = calc.get_homo_lumo(spin=1)
            dxc_pot = resp.calculate_discontinuity_potential((homoa, homob),
                                                             (lumoa, lumob))
            Eksa, Dxca = resp.calculate_discontinuity(dxc_pot, spin=0)
            Eksb, Dxcb = resp.calculate_discontinuity(dxc_pot, spin=1)
            Gapa = Eksa + Dxca
            Gapb = Eksb + Dxcb
            print("GAP", spin, Gapa, Gapb)
        else:
            homo, lumo = calc.get_homo_lumo()
            dxc_pot = resp.calculate_discontinuity_potential(homo, lumo)
            Eks, Dxc = resp.calculate_discontinuity(dxc_pot)
            Gap = Eks + Dxc
            print("GAP", spin, Gap)

    equal(Gapa, Gapb, 1e-4)
    equal(Gapa, Gap, 1e-4)
