# Test that LCAO wavefunctions are available and equal after restarts
# in normal as well as 'all' mode

import pytest
import numpy as np
from ase.build import molecule
from gpaw import GPAW


@pytest.mark.skip(reason='TODO')
def test_restart():
    system = molecule('H2')
    system.center(vacuum=2.5)

    calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.3, nbands=1, txt=None)
    system.calc = calc
    system.get_potential_energy()
    wf = calc.get_pseudo_wave_function(0)

    for mode in ['all', 'normal']:
        fname = 'lcao-restart.%s.gpw' % mode
        calc.write(fname, mode=dict(normal='', all='all')[mode])

        calc2 = GPAW(fname, txt=None)
        if mode == 'normal':
            continue
        wf2 = calc2.get_pseudo_wave_function(0)
        err = np.abs(wf2 - wf).max()
        print('%s: err=%s' % (mode, repr(err)))
        assert abs(err) < 1e-14
