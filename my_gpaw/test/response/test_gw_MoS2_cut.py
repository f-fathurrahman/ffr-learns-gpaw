import pytest
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from my_gpaw import GPAW, FermiDirac
from my_gpaw.response.g0w0 import G0W0


@pytest.fixture
def gpwfile(in_tmp_dir):
    calc = GPAW(
        mode='pw',
        xc='PBE',
        experimental={'niter_fixdensity': 2},
        nbands=16,
        convergence={'bands': 15},
        setups={'Mo': '6'},
        occupations=FermiDirac(0.001),
        kpts={'size': (3, 3, 1), 'gamma': True})

    a = 3.1604
    c = 10.0

    cell = Hexagonal(symbol='Mo',
                     latticeconstant={'a': a, 'c': c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=True,
                  scaled_positions=[(0, 0, 0),
                                    (2 / 3, 1 / 3, 0.3),
                                    (2 / 3, 1 / 3, -0.3)])

    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172 / 2
    pos[2][2] = pos[0][2] - 3.172 / 2
    layer.set_positions(pos)
    layer.calc = calc
    layer.get_potential_energy()
    fname = 'MoS2.gpw'
    calc.write(fname, mode='all')
    return fname


@pytest.mark.response
def test_response_gw_MoS2_cut(scalapack, gpwfile, needs_ase_master):
    gw = G0W0(gpwfile,
              'gw-test',
              nbands=15,
              ecut=10,
              eta=0.2,
              frequencies={'type': 'nonlinear', 'domega0': 0.1},
              truncation='2D',
              kpts=[((1 / 3, 1 / 3, 0))],
              bands=(8, 10))

    e_qp = gw.calculate()['qp'][0, 0]

    paths = gw.savepckl()
    for path in paths.values():
        assert path.exists()

    ev = 2.392
    ec = 7.337
    assert e_qp[0] == pytest.approx(ev, abs=0.01)
    assert e_qp[1] == pytest.approx(ec, abs=0.01)
