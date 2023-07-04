from ase import Atoms
from gpaw import GPAW
from gpaw.test import equal
from gpaw.xc.tools import vxc


def test_xc_degeneracy():
    a = 5.0
    d = 1.0
    x = d / 3**0.5
    atoms = Atoms('CH4',
                  [(0.0, 0.0, 0.0),
                   (x, x, x),
                   (-x, -x, x),
                   (x, -x, -x),
                   (-x, x, -x)],
                  cell=(a, a, a),
                  pbc=False)

    atoms.positions[:] += a / 2
    params = dict(h=0.25, nbands=4, convergence={'eigenstates': 7.8e-10})
    atoms.calc = GPAW(**params)
    energy = atoms.get_potential_energy()

    # The three eigenvalues e[1], e[2], and e[3] must be degenerate:
    e = atoms.calc.get_eigenvalues()
    print(e[1] - e[3])
    equal(e[1], e[3], 9.3e-8)

    energy_tolerance = 0.002
    equal(energy, -23.631, energy_tolerance)

    gs = atoms.calc.gs_adapter()

    # Calculate non-selfconsistent PBE eigenvalues:
    epbe0 = e[:2] - vxc(gs, n2=2)[0, 0] + vxc(gs, 'PBE', n2=2)[0, 0]

    # Calculate selfconsistent PBE eigenvalues:
    atoms.calc = GPAW(**params, xc='PBE')
    energy = atoms.get_potential_energy()
    epbe = atoms.calc.get_eigenvalues()

    de = epbe[1] - epbe[0]
    de0 = epbe0[1] - epbe0[0]
    print(de, de0)
    equal(de, de0, 0.001)
