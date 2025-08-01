import numpy as np
import pytest
from ase import Atoms
from my_gpaw25.test import calculate_numerical_forces

import my_gpaw25.cgpaw as cgpaw
from my_gpaw25 import GPAW, Mixer, PoissonSolver
from my_gpaw25.external import PointChargePotential


@pytest.mark.old_gpaw_only
def test_ext_potential_point_charge(in_tmp_dir):
    # Find coefs for polynomial:
    c = np.linalg.solve([[1, 1, 1, 1],
                         [0, 2, 4, 6],
                         [0, 2, 12, 30],
                         [1 / 3, 1 / 5, 1 / 7, 1 / 9]],
                        [1, -1, 2, 0.5])
    print(c)
    print(c * 32)
    x = np.linspace(0, 1, 101)
    v = np.polyval(c[::-1], x**2)
    assert (v * x**2 - x).sum() / 100 == pytest.approx(0, abs=1e-5)

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, v)
        x = np.linspace(0.2, 1.5, 101)
        plt.plot(x, 1 / x)
        plt.show()

    # Low-level test:
    h = 0.1
    q = 2.2
    beg_v = np.zeros(3, int)
    h_v = np.ones(3) * h
    q_p = np.ones(1) * q
    R_pv = np.array([[0.1, 0.2, -0.3]])
    vext_G = np.zeros((1, 1, 1))
    rhot_G = np.ones((1, 1, 1))

    def f(rc):
        vext_G[:] = 0.0
        cgpaw.pc_potential(beg_v, h_v, q_p, R_pv, rc,
                           np.inf, 1.0, vext_G, None)
        return -vext_G[0, 0, 0]

    d = (R_pv[0]**2).sum()**0.5

    for rc in [0.9 * d, 1.1 * d]:
        if d > rc:
            v0 = q / d
        else:
            v0 = np.polyval(c[::-1], (d / rc)**2) * q / rc

        assert f(rc) == pytest.approx(v0, abs=1e-12)

        F_pv = np.zeros((1, 3))
        cgpaw.pc_potential(beg_v, h_v, q_p, R_pv, rc, np.inf, 1.0,
                           vext_G, None, rhot_G, F_pv)
        eps = 0.0001
        for v in range(3):
            R_pv[0, v] += eps
            ep = f(rc)
            R_pv[0, v] -= 2 * eps
            em = f(rc)
            R_pv[0, v] += eps
            F = -(ep - em) / (2 * eps) * h**3
            assert F == pytest.approx(-F_pv[0, v], abs=1e-9)

    # High-level test:
    lih = Atoms('LiH')
    lih[1].y += 1.64
    lih.center(vacuum=3)

    pos = lih.cell.sum(axis=0)
    print(pos)
    pc = PointChargePotential([-1.0], [pos])
    lih.calc = GPAW(mode='fd',
                    external=pc,
                    mixer=Mixer(0.8, 5, 20.0),
                    xc='oldLDA',
                    poissonsolver=PoissonSolver(),
                    convergence=dict(density=3e-6),
                    txt='lih-pc.txt')
    f1 = lih.get_forces()
    fpc1 = pc.get_forces(lih.calc)
    print(fpc1)
    print(fpc1 + f1.sum(0))

    f2 = calculate_numerical_forces(lih, 0.001)
    print(f1)
    print(f1 - f2)
    err = abs(f1 - f2).max()
    print('err', err)
    assert err < 2e-3, err

    x = 0.0001
    for v in range(3):
        pos[v] += x
        pc.set_positions([pos])
        ep = lih.get_potential_energy()
        pos[v] -= 2 * x
        pc.set_positions([pos])
        em = lih.get_potential_energy()
        pos[v] += x
        pc.set_positions([pos])
        error = (em - ep) / (2 * x) - fpc1[0, v]
        print(v, error)
        assert abs(error) < 0.006
