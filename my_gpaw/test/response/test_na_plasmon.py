import pytest
import numpy as np

from ase import Atoms
from my_gpaw import GPAW, PW
from my_gpaw.mpi import world
from my_gpaw.test import equal, findpeak
from my_gpaw.utilities import compiled_with_sl
from my_gpaw.response.df import DielectricFunction

# Comparing the plasmon peaks found in bulk sodium for two different
# atomic structures. Testing for idential plasmon peaks. Not using
# physical sodium cell.


@pytest.mark.response
def test_response_na_plasmon(in_tmp_dir):
    a = 4.23 / 2.0
    a1 = Atoms('Na',
               scaled_positions=[[0, 0.1, 0]],
               cell=(a, a, a),
               pbc=True)

    # Expanding along x-direction
    a2 = Atoms('Na2',
               scaled_positions=[[0, 0.1, 0], [0.5, 0.1, 0]],
               cell=(2 * a, a, a),
               pbc=True)

    a1.calc = GPAW(mode=PW(250),
                   kpts={'size': (8, 8, 8), 'gamma': True},
                   parallel={'band': 1},
                   # txt='small.txt',
                   )

    # Kpoint sampling should be halved in the expanded direction.
    a2.calc = GPAW(mode=PW(250),
                   kpts={'size': (4, 8, 8), 'gamma': True},
                   parallel={'band': 1},
                   # txt='large.txt',
                   )

    a1.get_potential_energy()
    a2.get_potential_energy()

    # Use twice as many bands for expanded structure
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a2.calc.diagonalize_full_hamiltonian(nbands=40)

    a1.calc.write('gs_Na_small.gpw', 'all')
    a2.calc.write('gs_Na_large.gpw', 'all')

    # Settings that should yield the same result
    settings = [{'disable_point_group': True, 'disable_time_reversal': True},
                {'disable_point_group': False, 'disable_time_reversal': True},
                {'disable_point_group': True, 'disable_time_reversal': False},
                {'disable_point_group': False, 'disable_time_reversal': False}]

    # Test block parallelization (needs scalapack)
    if world.size > 1 and compiled_with_sl():
        settings.append({'disable_point_group': False,
                         'disable_time_reversal': False,
                         'nblocks': 2})

    # Calculate the dielectric functions
    dfs0 = []  # Arrays to check for self-consistency
    dfs1 = []
    dfs2 = []
    dfs3 = []
    dfs4 = []
    dfs5 = []
    for kwargs in settings:
        df1 = DielectricFunction('gs_Na_small.gpw',
                                 ecut=40,
                                 rate=0.001,
                                 **kwargs)

        df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
        df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
        df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')

        df2 = DielectricFunction('gs_Na_large.gpw',
                                 ecut=40,
                                 rate=0.001,
                                 **kwargs)

        df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
        df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
        df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')

        dfs0.append(df1NLFCx)
        dfs1.append(df1LFCx)
        dfs2.append(df1NLFCy)
        dfs3.append(df1LFCy)
        dfs4.append(df1NLFCz)
        dfs5.append(df1LFCz)

        # Compare plasmon frequencies and intensities
        w_w = df1.wd.omega_w
        w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
        equal(w1, w2, 1e-2)
        equal(I1, I2, 1e-3)

        w1, I1 = findpeak(w_w, -(1. / df1LFCy).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCy).imag)
        equal(w1, w2, 1e-2)
        equal(I1, I2, 1e-3)

        w1, I1 = findpeak(w_w, -(1. / df1LFCz).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCz).imag)
        equal(w1, w2, 1e-2)
        equal(I1, I2, 1e-3)

    # Check for self-consistency
    for i, dfs in enumerate([dfs0, dfs1, dfs2, dfs3, dfs4, dfs5]):
        while len(dfs):
            df = dfs.pop()
            for df2 in dfs:
                try:
                    assert np.max(np.abs((df - df2) / df)) < 1e-3
                except AssertionError:
                    print(np.max(np.abs((df - df2) / df)))
                    raise AssertionError
