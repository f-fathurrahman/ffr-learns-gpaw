import pytest
from ase import Atom
from my_gpaw import GPAW
from my_gpaw.cluster import Cluster
from my_gpaw.test import equal


@pytest.mark.mgga
def test_mgga_mgga_restart(in_tmp_dir):
    fname = 'H2_PBE.gpw'
    fwfname = 'H2_wf_PBE.gpw'
    txt = None

    # write first if needed
    try:
        c = GPAW(fname, txt=txt)
        c = GPAW(fwfname, txt=txt)
    except FileNotFoundError:
        s = Cluster([Atom('H'), Atom('H', [0, 0, 1])])
        s.minimal_box(3.)
        c = GPAW(xc={'name': 'PBE', 'stencil': 1},
                 h=.3, convergence={'density': 1e-4, 'eigenstates': 1e-6})
        c.calculate(s)
        c.write(fname)
        c.write(fwfname, 'all')

    # full information
    c = GPAW(fwfname, txt=txt)
    E_PBE = c.get_potential_energy()
    dE = c.get_xc_difference({'name': 'TPSS', 'stencil': 1})
    E_1 = E_PBE + dE
    print('E PBE, TPSS=', E_PBE, E_1)

    # no wfs
    c = GPAW(fname, txt=txt)
    E_PBE_no_wfs = c.get_potential_energy()
    dE = c.get_xc_difference({'name': 'TPSS', 'stencil': 1})
    E_2 = E_PBE_no_wfs + dE
    print('E PBE, TPSS=', E_PBE_no_wfs, E_2)

    print('diff=', E_1 - E_2)
    assert abs(E_1 - E_2) < 0.005

    energy_tolerance = 0.002
    equal(E_PBE, -5.341, energy_tolerance)
    equal(E_PBE_no_wfs, -5.33901, energy_tolerance)
    equal(E_1, -5.57685, energy_tolerance)
    equal(E_2, -5.57685, energy_tolerance)
