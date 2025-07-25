import os
from contextlib import contextmanager

import numpy as np
import pytest
from my_gpaw25 import setup_paths, GPAW_NEW
from my_gpaw25.cli.info import info
from my_gpaw25.mpi import broadcast, world
from my_gpaw25.test.gpwfile import GPWFiles, _all_gpw_methodnames
from my_gpaw25.test.mmefile import MMEFiles
from my_gpaw25.utilities import devnull


@contextmanager
def execute_in_tmp_path(request, tmp_path_factory):
    if world.rank == 0:
        # Obtain basename as
        # * request.function.__name__  for function fixture
        # * request.module.__name__    for module fixture
        basename = getattr(request, request.scope).__name__
        path = tmp_path_factory.mktemp(basename)
    else:
        path = None
    path = broadcast(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='function')
def in_tmp_dir(request, tmp_path_factory):
    """Run test function in a temporary directory."""
    with execute_in_tmp_path(request, tmp_path_factory) as path:
        yield path


@pytest.fixture(scope='module')
def module_tmp_path(request, tmp_path_factory):
    """Run test module in a temporary directory."""
    with execute_in_tmp_path(request, tmp_path_factory) as path:
        yield path


@pytest.fixture
def add_cwd_to_setup_paths():
    """Temporarily add current working directory to setup_paths."""
    try:
        setup_paths[:0] = ['.']
        yield
    finally:
        del setup_paths[:1]


@pytest.fixture(scope='session')
def sessionscoped_monkeypatch():
    # The standard monkeypatch fixture is function scoped
    # so we need to roll our own
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield monkeypatch


@pytest.fixture(autouse=True, scope='session')
def monkeypatch_response_spline_points(sessionscoped_monkeypatch):
    import my_gpaw25.response.paw as paw
    # https://gitlab.com/gpaw/gpaw/-/issues/984
    sessionscoped_monkeypatch.setattr(paw, 'DEFAULT_RADIAL_POINTS', 2**10)


@pytest.fixture(scope='session')
def gpw_files(request):
    """Reuse gpw-files.

    Returns a dict mapping names to paths to gpw-files.
    The files are written to the pytest cache and can be cleared using
    pytest --cache-clear.

    Example::

        def test_something(gpw_files):
            calc = GPAW(gpw_files['h2_lcao'])
            ...

    Possible systems are:

    * Bulk BCC-Li with 3x3x3 k-points: ``bcc_li_pw``, ``bcc_li_fd``,
      ``bcc_li_lcao``.

    * O2 molecule: ``o2_pw``.

    * H2 molecule: ``h2_pw``, ``h2_fd``, ``h2_lcao``.

    * H2 molecule (not centered): ``h2_pw_0``.

    * N2 molecule ``n2_pw``

    * N molecule ``n_pw``

    * Spin-polarized H atom: ``h_pw``.

    * Polyethylene chain.  One unit, 3 k-points, no symmetry:
      ``c2h4_pw_nosym``.  Three units: ``c6h12_pw``.

    * Bulk BN (zinkblende) with 2x2x2 k-points and 9 converged bands:
      ``bn_pw``.

    * h-BN layer with 3x3x1 (gamma center) k-points and 26 converged bands:
      ``hbn_pw``.

    * Graphene with 6x6x1 k-points: ``graphene_pw``

    * I2Sb2 (Z2 topological insulator) with 6x6x1 k-points and no
      symmetries: ``i2sb2_pw_nosym``

    * MoS2 with 6x6x1 k-points: ``mos2_pw`` and ``mos2_pw_nosym``

    * MoS2 with 5x5x1 k-points: ``mos2_5x5_pw``

    * NiCl2 with 6x6x1 k-points: ``nicl2_pw`` and ``nicl2_pw_evac``

    * V2Br4 (AFM monolayer), LDA, 4x2x1 k-points, 28(+1) converged bands:
      ``v2br4_pw`` and ``v2br4_pw_nosym``

    * Bulk Si, LDA, 2x2x2 k-points (gamma centered): ``si_pw``

    * Bulk Si, LDA, 4x4x4 k-points, 8(+1) converged bands: ``fancy_si_pw``
      and ``fancy_si_pw_nosym``

    * Bulk SiC, LDA, 4x4x4 k-points, 8(+1) converged bands: ``sic_pw``
      and ``sic_pw_spinpol``

    * Bulk Fe, LDA, 4x4x4 k-points, 9(+1) converged bands: ``fe_pw``
      and ``fe_pw_nosym``

    * Bulk C, LDA, 2x2x2 k-points (gamma centered), ``c_pw``

    * Bulk Co (HCP), 4x4x4 k-points, 12(+1) converged bands: ``co_pw``
      and ``co_pw_nosym``

    * Bulk SrVO3 (SC), 3x3x3 k-points, 20(+1) converged bands: ``srvo3_pw``
      and ``srvo3_pw_nosym``

    * Bulk Al, LDA, 4x4x4 k-points, 10(+1) converged bands: ``al_pw``
      and ``al_pw_nosym``

    * Bulk Al, LDA, 4x4x4 k-points, 4 converged bands: ``bse_al``

    * Bulk Ag, LDA, 2x2x2 k-points, 6 converged bands,
      2eV U on d-band: ``ag_pw``

    * Bulk GaAs, LDA, 4x4x4 k-points, 8(+1) bands converged: ``gaas_pw``
      and ``gaas_pw_nosym``

    * Bulk P4, LDA, 4x4 k-points, 40 bands converged: ``p4_pw``

    * Distorted bulk Fe, revTPSS: ``fe_pw_distorted``

    * Distorted bulk Si, TPSS: ``si_pw_distorted``

    Files always include wave functions.
    """
    cache = request.config.cache
    gpaw_cachedir = cache.mkdir('gpaw_test_gpwfiles')

    gpwfiles = GPWFiles(gpaw_cachedir)

    try:
        setup_paths.append(gpwfiles.testing_setup_path)
        yield gpwfiles
    finally:
        setup_paths.remove(gpwfiles.testing_setup_path)


@pytest.fixture(scope='session', params=sorted(_all_gpw_methodnames))
def all_gpw_files(request, gpw_files, pytestconfig):
    """This fixture parametrizes a test over all gpw_files.

    For example pytest test_generate_gpwfiles.py -n 16 is a way to quickly
    generate all gpw files independently of the rest of the test suite."""

    # Note: Parametrizing over _all_gpw_methodnames must happen *after*
    # it is populated, i.e., further down in the file than
    # the @gpwfile decorator.

    # TODO This xfail-information should probably live closer to the
    # gpwfile definitions and not here in the fixture.
    skip_if_new = {'Cu3Au_qna',
                   'nicl2_pw', 'nicl2_pw_evac',
                   'v2br4_pw', 'v2br4_pw_nosym',
                   'sih4_xc_gllbsc_fd', 'sih4_xc_gllbsc_lcao',
                   'na2_isolated', 'h2o_xas'}
    if GPAW_NEW and request.param in skip_if_new:
        pytest.xfail(f'{request.param} gpwfile not yet working with GPAW_NEW')

    # Accessing each file via __getitem__ executes the calculation:
    return gpw_files[request.param]


@pytest.fixture(scope='session')
def mme_files(request, gpw_files):
    """Reuse mme files"""
    cache = request.config.cache
    mme_cachedir = cache.mkdir('gpaw_test_mmefiles')

    return MMEFiles(mme_cachedir, gpw_files)


class GPAWPlugin:
    def __init__(self):
        if world.rank == -1:
            print()
            info()

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from my_gpaw25.mpi import size
        terminalreporter.section('GPAW-MPI stuff')
        terminalreporter.write(f'size: {size}\n')


@pytest.fixture
def sg15_hydrogen():
    from io import StringIO
    from my_gpaw25.test.pseudopotential.H_sg15 import pp_text
    from my_gpaw25.upf import read_sg15
    # We can't easily load a non-python file from the test suite.
    # Therefore we load the pseudopotential from a Python file.
    return read_sg15(StringIO(pp_text))


def pytest_configure(config):
    # Allow for fake cupy:
    os.environ['GPAW_CPUPY'] = '1'

    if world.rank != 0:
        try:
            tw = config.get_terminal_writer()
        except (AssertionError, AttributeError):
            pass
        else:
            tw._file = devnull
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')


def pytest_runtest_setup(item):
    """Skip some tests.

    If:

    * they depend on libxc and GPAW is not compiled with libxc
    * they are before $PYTEST_START_AFTER
    """
    from my_gpaw25 import get_libraries
    libraries = get_libraries()

    if world.size > 1:
        for mark in item.iter_markers():
            if mark.name == 'serial':
                pytest.skip('Only run in serial')
    else:
        for mark in item.iter_markers():
            if mark.name == 'parallel':
                pytest.skip('Only run in parallel')

    if item.location[0] <= os.environ.get('PYTEST_START_AFTER', ''):
        pytest.skip('Not after $PYTEST_START_AFTER')
        return

    if libraries['libxc']:
        return

    if any(mark.name in {'libxc', 'mgga'}
           for mark in item.iter_markers()):
        pytest.skip('No LibXC.')


@pytest.fixture
def scalapack():
    """Skip if not compiled with sl.

    This fixture otherwise does not return or do anything."""
    from my_gpaw25.utilities import compiled_with_sl
    if not compiled_with_sl():
        pytest.skip('no scalapack')


@pytest.fixture
def needs_ase_master():
    from ase.utils.filecache import MultiFileJSONCache
    try:
        MultiFileJSONCache('bla-bla', comm=None)
    except TypeError:
        pytest.skip('ASE is too old')


def pytest_report_header(config, start_path):
    # Use this to add custom information to the pytest printout.
    yield f'GPAW MPI rank={world.rank}, size={world.size}'

    # We want the user to be able to see where gpw files are cached,
    # but the only way to see the cache location is to make a directory
    # inside it.  mkdir('') returns the toplevel cache dir without
    # actually creating a subdirectory:
    cachedir = config.cache.mkdir('')
    yield f'Cache directory including gpw files: {cachedir}'


@pytest.fixture
def rng():
    """Seeded random number generator.

    Tests should be deterministic and should use this
    fixture or initialize their own rng."""
    return np.random.default_rng(42)


@pytest.fixture
def gpaw_new() -> bool:
    """Are we testing the new code?"""
    return GPAW_NEW
