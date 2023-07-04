import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac, PW
from gpaw.response.df import DielectricFunction
from gpaw.test import equal, findpeak


@pytest.fixture
def gpwfile(in_tmp_dir):
    cluster = Atoms('Au2', [(0, 0, 0), (0, 0, 2.564)])
    cluster.set_cell((6, 6, 6), scale_atoms=False)
    cluster.center()
    calc = GPAW(mode=PW(ecut=180, force_complex_dtype=True),
                xc={'name': 'RPBE', 'stencil': 1},
                nbands=16,
                eigensolver='rmm-diis',
                parallel={'domain': 1},
                occupations=FermiDirac(0.01))

    cluster.calc = calc
    cluster.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=24, scalapack=True)
    gpwname = 'Au2.gpw'
    calc.write(gpwname, 'all')
    return gpwname


@pytest.mark.response
@pytest.mark.slow
def test_response_au02_absorption(scalapack, in_tmp_dir, gpwfile):
    nw = 141
    frequencies = np.linspace(0, 14, nw)

    df = DielectricFunction(gpwfile,
                            frequencies=frequencies,
                            hilbert=False,
                            eta=0.1,
                            ecut=10)

    b0, b = df.get_dielectric_function(filename=None,
                                       direction='z')
    a0, a = df.get_polarizability(filename=None,
                                  direction='z')

    w0_ = 5.60491055
    I0_ = 227.23392824591642
    w_ = 5.644900254787107
    I_ = 184.4086028397282

    w, I = findpeak(frequencies, b0.imag)
    equal(w, w0_, 0.05)
    equal(6**3 * I / (4 * np.pi), I0_, 0.5)
    w, I = findpeak(frequencies, a0.imag)
    equal(w, w0_, 0.05)
    equal(I, I0_, 0.5)
    w, I = findpeak(frequencies, b.imag)
    equal(w, w_, 0.05)
    equal(6**3 * I / (4 * np.pi), I_, 0.5)
    w, I = findpeak(frequencies, a.imag)
    equal(w, w_, 0.05)
    equal(I, I_, 0.5)
