import numpy as np
import pytest

import my_gpaw25.mpi as mpi
from my_gpaw25.grid_descriptor import GridDescriptor
from my_gpaw25.kpt_descriptor import KPointDescriptor
from my_gpaw25.lfc import LocalizedFunctionsCollection as LFC
from my_gpaw25.pw.descriptor import PWDescriptor
from my_gpaw25.pw.lfc import PWLFC
from my_gpaw25.spline import Spline


@pytest.mark.ci
def test_pw_lfc():
    x = 2.0
    rc = 3.5
    r = np.linspace(0, rc, 100)

    n = 40
    a = 8.0
    gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)

    kpts = np.array([(0.25, 0.25, 0.0)])
    kd = KPointDescriptor(kpts)
    spos_ac = np.array([(0.15, 0.5, 0.95)])

    pd = PWDescriptor(45, gd, complex, kd)

    eikr = np.ascontiguousarray(
        np.exp(2j * np.pi * np.dot(np.indices(gd.N_c).T,
                                   (kpts / gd.N_c).T).T)[0])

    for l in range(3):
        print(l)
        s = Spline.from_data(l, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))

        lfc1 = LFC(gd, [[s]], kd, dtype=complex)
        lfc2 = PWLFC([[s]], pd)

        c_axi = {0: np.zeros((1, 2 * l + 1), complex)}
        c_axi[0][0, 0] = 1.9 - 4.5j
        c_axiv = {0: np.zeros((1, 2 * l + 1, 3), complex)}

        b1 = gd.zeros(1, dtype=complex)
        b2 = pd.zeros(1, dtype=complex)

        for lfc, b in [(lfc1, b1), (lfc2, b2)]:
            lfc.set_positions(spos_ac)
            lfc.add(b, c_axi, 0)

        b2 = pd.ifft(b2[0]) * eikr
        assert abs(b2 - b1[0]).max() == pytest.approx(0, abs=0.001)

        b1 = eikr[None]
        b2 = pd.fft(b1[0] * 0 + 1).reshape((1, -1))

        results = []
        results2 = []
        for lfc, b in [(lfc1, b1), (lfc2, b2)]:
            lfc.integrate(b, c_axi, 0)
            results.append(c_axi[0][0].copy())
            lfc.derivative(b, c_axiv, 0)
            results2.append(c_axiv[0][0].copy())
        assert abs(np.ptp(results2, 0)).max() == pytest.approx(0, abs=1e-7)
        assert abs(np.ptp(results, 0)).max() == pytest.approx(0, abs=3e-8)
