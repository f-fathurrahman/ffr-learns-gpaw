import pytest
from ase import Atoms
from ase.units import Ha

from my_gpaw import SCIPY_VERSION
from my_gpaw.new.calculation import DFTCalculation
from my_gpaw.mpi import size


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.skipif(SCIPY_VERSION < [1, 6], reason='Too old scipy')
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('gpu', [False, True])
def test_gpu_pw(dtype, gpu):
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             dtype=dtype,
             parallel={'gpu': gpu},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    energy = dft.results['energy'] * Ha
    assert energy == pytest.approx(-16.032945, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.skipif(SCIPY_VERSION < [1, 6], reason='Too old scipy')
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('par', ['domain', 'kpt'])
def test_gpu_pw_k(gpu, par):
    atoms = Atoms('H', pbc=True, cell=[1, 1, 1])
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             kpts=(4, 1, 1),
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    dft.forces()
    if par == 'kpt':
        dft.stress()
    energy = dft.results['energy'] * Ha
    assert energy == pytest.approx(-19.579937, abs=1e-6)
