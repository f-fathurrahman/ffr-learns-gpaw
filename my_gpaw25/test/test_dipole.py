"""
Test the dipole correction code by comparing this system:

    H
z1   O  z2
    H


(where z1 and z2 denote points where the potential is probed)

Expected potential:

      -----
     /
    /
----

to this system:

   H           H
z1  O    z2   O
   H           H


Expected potential:

       -------
      /       \
     /         \
-----           ------

The height of the two potentials are tested to be the same.

Enable if-statement in the bottom for nice plots
"""
import numpy as np
from ase.build import molecule
from ase.units import Hartree
from my_gpaw25 import GPAW, Mixer
from my_gpaw25.mpi import rank


def test_dipole():
    system1 = molecule('H2O')
    system1.set_pbc((True, True, False))
    system1.cell = 4.0 * np.array([[1.0, -1.5, 0.0], [1.0, 1.0, 0.0],
                                   [0., 0., 1.]])
    system1.center(vacuum=10.0, axis=2)

    system2 = system1.copy()
    system2.positions *= [1.0, 1.0, -1.0]
    system2 += system1
    system2.center(vacuum=6.0, axis=2)

    convergence = dict(density=1e-6)

    def kw():
        return dict(mode='lcao',
                    convergence=convergence,
                    mixer=Mixer(0.5, 7, 50.0),
                    h=0.25,
                    xc='oldLDA')

    calc1 = GPAW(poissonsolver={'dipolelayer': 'xy'}, **kw())

    system1.calc = calc1
    system1.get_potential_energy()
    v1 = calc1.get_effective_potential()

    calc2 = GPAW(**kw())

    system2.calc = calc2
    system2.get_potential_energy()
    v2 = calc2.get_effective_potential()

    def get_avg(v):
        nx, ny, nz = v.shape
        vyz = v.sum(axis=0) / nx
        vz = vyz.sum(axis=0) / ny
        return vz, vyz

    if rank == 0:
        vz1, vyz1 = get_avg(v1)
        vz2, vyz2 = get_avg(v2)

        # Compare values that are not quite at the end of the array
        # (at the end of the array things can "oscillate" a bit)
        dvz1 = vz1[-3] - vz1[3]
        dvz2 = vz2[3] - vz2[len(vz2) // 2]
        print(dvz1, dvz2)

        err1 = abs(dvz1 - dvz2)

        try:
            correction = calc1.hamiltonian.poisson.correction
        except AttributeError:
            correction = (calc1.dft.pot_calc.poisson_solver.
                          solver.correction)

        correction_err = abs(2.0 * correction * Hartree + dvz1)
        print('correction error %s' % correction_err)
        assert correction_err < 3e-5

        # Comparison to what the values were when this test was last modified:
        ref_value = 2.07342988218
        err2 = abs(dvz1 - ref_value)

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(vyz1)
            plt.figure()
            plt.imshow(vyz2)
            plt.figure()
            plt.plot(vz1)
            plt.plot(vz2)
            plt.show()

        print('Ref value of previous calculation', ref_value)
        print('Value in this calculation', dvz1)

        # fine grid needed to achieve convergence!
        print('Error', err1, err2)
        assert err1 < 4e-3, err1
        assert err2 < 2e-4, err2
