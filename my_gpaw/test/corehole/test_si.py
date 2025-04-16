import pytest
from ase import Atoms

from my_gpaw import GPAW, FermiDirac
from my_gpaw.test import gen
from my_gpaw.xas import XAS, RecursionMethod


@pytest.mark.later
def test_corehole_si(in_tmp_dir, add_cwd_to_setup_paths):
    # Generate setup for oxygen with half a core-hole:
    gen('Si', name='hch1s', corehole=(1, 0, 0.5), gpernode=30, write_xml=True)
    a = 2.6
    si = Atoms('Si', cell=(a, a, a), pbc=True)

    import numpy as np
    calc = GPAW(nbands=None,
                h=0.25,
                occupations=FermiDirac(width=0.05),
                setups='hch1s',
                convergence={'maximum iterations': 1})
    si.calc = calc
    _ = si.get_potential_energy()
    calc.write('si.gpw')

    # restart from file
    calc = GPAW('si.gpw')

    import gpaw.mpi as mpi
    if mpi.size == 1:
        xas = XAS(calc)
        x, y = xas.get_spectra()
    else:
        x = np.linspace(0, 10, 50)

    k = 2
    calc = calc.new(kpts=(k, k, k))
    calc.initialize(si)
    calc.set_positions(si)
    assert calc.wfs.dtype == complex

    r = RecursionMethod(calc)
    r.run(40)
    if mpi.size == 1:
        z = r.get_spectra(x)

    if 0:
        import pylab as p
        p.plot(x, y[0])
        p.plot(x, sum(y))
        p.plot(x, z[0])
        p.show()

    # 2p corehole
    s = gen('Si', name='hch2p', corehole=(2, 1, 0.5), gpernode=30)
    calc = GPAW(nbands=None,
                h=0.25,
                occupations=FermiDirac(width=0.05),
                setups={0: s})
    si.calc = calc

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 1)
    _ = si.get_potential_energy()
