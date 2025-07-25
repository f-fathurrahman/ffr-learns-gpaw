from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import IO, Any, Callable, Protocol, Sequence, Union, Iterable

import numpy as np
from ase import Atoms
from ase.units import Ha
from my_gpaw25 import __version__
from my_gpaw25.core import UGArray
from my_gpaw25.dos import DOSCalculator
from my_gpaw25.mpi import MPIComm
from my_gpaw25.mpi import synchronize_atoms, world
from my_gpaw25.new import Timer, trace
from my_gpaw25.new.builder import builder as create_builder
from my_gpaw25.new.calculation import (CalculationModeError, DFTCalculation,
                                  ReuseWaveFunctionsError, units)
from my_gpaw25.new.gpw import read_gpw, write_gpw
from my_gpaw25.new.input_parameters import InputParameters
from my_gpaw25.new.input_parameters import parameter_functions as parameter_names
from my_gpaw25.new.logger import Logger
from my_gpaw25.new.pw.fulldiag import diagonalize
from my_gpaw25.new.xc import create_functional
from my_gpaw25.typing import Array1D, Array2D, Array3D
from my_gpaw25.utilities import pack_density
from my_gpaw25.utilities.memory import maxrss


class Dictable(Protocol):
    def todict(self) -> dict[str, Any]:
        ...


def GPAW(
    filename: Union[str, Path, IO[str]] = None,
    *,
    txt: str | Path | IO[str] | None = '?',
    communicator: MPIComm | Iterable[int] | None = None,
    basis: str | dict[str | int | None, str] | None = None,
    charge: float | None = None,
    convergence: dict[str, Any] | None = None,
    eigensolver: dict[str, Any] | None = None,
    experimental: dict[str, Any] | None = None,
    external: dict[str, Any] | None = None,
    gpts: None | Sequence[int] | None = None,
    h: float | None = None,
    hund: bool | None = None,
    kpts: dict[str, Any] | None = None,
    magmoms: Any | None = None,
    maxiter: int | None = None,
    mixer: dict[str, Any] | None = None,
    mode: str | dict[str, Any] | None = None,
    nbands: int | str | None = None,
    occupations: dict[str, Any] | None = None,
    parallel: dict[str, Any] | None = None,
    poissonsolver: dict[str, Any] | None = None,
    random: bool | None = None,
    setups: Any | None = None,
    soc: bool | None = None,
    spinpol: bool | None = None,
    symmetry: str | dict[str, Any] | None = None,
    xc: str | dict[str, Any] | Dictable | None = None) -> ASECalculator:

    """Create ASE-compatible GPAW calculator.

    """
    if txt == '?':
        txt = '-' if filename is None else None

    if communicator is None:
        comm = world
    elif not hasattr(communicator, 'rank'):
        comm = world.new_communicator(list(communicator))
    else:
        comm = communicator  # type: ignore

    log = Logger(txt, comm)

    params_dict = {key: value for key, value in locals().items()
                   if key in parameter_names}

    if filename is not None:
        for key, value in params_dict.items():
            if key != 'parallel' and value is not None:
                raise ValueError(
                    f'Illegal argument when reading from a file: {key}')
        atoms, dft, params, _ = read_gpw(filename,
                                         log=log,
                                         parallel=parallel)
        return ASECalculator(params,
                             log=log, dft=dft, atoms=atoms)

    params = InputParameters(params_dict)
    write_header(log, params)
    return ASECalculator(params, log=log)


LOGO = """\
  ___ ___ ___ _ _ _
 |   |   |_  | | | |
 | | | | | . | | | |
 |__ |  _|___|_____| - {version}
 |___|_|
"""


def write_header(log, params):
    from my_gpaw25.io.logger import write_header as header
    log(LOGO.format(version=__version__))
    header(log, log.comm)
    with log.indent('input parameters:'):
        parts = []
        for key, val in params.items():
            n = len(key)
            txt = pformat(val, width=75 - n).replace('\n', '\n ' + ' ' * n)
            parts.append(f'{key}={txt}')
        log(',\n'.join(parts))


def compare_atoms(a1: Atoms, a2: Atoms) -> set[str]:
    if a1 is a2:
        return set()

    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'numbers'}

    if (a1.pbc != a2.pbc).any():
        return {'pbc'}

    if abs(a1.cell - a2.cell).max() > 0.0:
        return {'cell'}

    if abs(a1.positions - a2.positions).max() > 0.0:
        return {'positions'}

    return set()


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""

    name = 'gpaw'
    old = False

    def __init__(self,
                 params: InputParameters,
                 *,
                 log: Logger,
                 dft: DFTCalculation | None = None,
                 atoms: Atoms | None = None):
        
        print("\n<div> ENTER New.ASECalculator.__init__\n")
        self.params = params
        self.log = log
        self.comm = log.comm
        self._dft = dft
        self._atoms = atoms
        self.timer = Timer()
        self.hooks: dict[str, Callable] = {}
        print("\n</div> EXIT New.ASECalculator.__init__\n")


    @property
    def dft(self) -> DFTCalculation:
        if self._dft is None:
            raise AttributeError
        return self._dft

    @property
    def atoms(self) -> Atoms:
        if self._atoms is None:
            raise AttributeError
        return self._atoms

    def __repr__(self):
        params = []
        for key, value in self.params.items():
            val = repr(value)
            if len(val) > 40:
                val = '...'
            params.append((key, val))
        p = ', '.join(f'{key}: {val}' for key, val in params)
        return f'ASECalculator({p})'

    def iconverge(self, atoms: Atoms | None):
        """Iterate to self-consistent solution.

        Will also calculate "cheap" properties: energy, magnetic moments
        and dipole moment.
        """

        print("\n<div> ENTER New.ASECalculator.iconverge\n")

        if atoms is None:
            atoms = self.atoms
        else:
            synchronize_atoms(atoms, self.comm)

        converged = True

        if self._dft is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'numbers', 'pbc', 'cell'}:
                if 'numbers' not in changes:
                    # Remember magmoms if there are any:
                    magmom_a = self.dft.results.get('magmoms')
                    if magmom_a is not None and magmom_a.any():
                        atoms = atoms.copy()
                        assert atoms is not None  # MYPY: why is this needed?
                        atoms.set_initial_magnetic_moments(magmom_a)

                if changes & {'numbers', 'pbc'}:
                    self._dft = None  # start from scratch
                else:
                    try:
                        self.create_new_calculation_from_old(atoms)
                    except ReuseWaveFunctionsError:
                        self._dft = None  # start from scratch
                    else:
                        converged = False
                        changes = set()

        if self._dft is None:
            print("---- Creating new calculation")
            self.create_new_calculation(atoms)
            converged = False
        elif changes:
            self.move_atoms(atoms)
            converged = False
        elif not self._dft.results:
            # Something cleared the results dict
            converged = False

        if converged:
            return

        if not self.dft.ibzwfs.has_wave_functions():
            self.create_new_calculation(atoms)

        assert self.hooks.keys() <= {'scf_step', 'converged'}

        with self.timer('SCF'):
            for ctx in self.dft.iconverge(
                    calculate_forces=self._calculate_forces):
                yield ctx
                self.hooks.get('scf_step', lambda ctx: None)(ctx)

        self.log(f'Converged in {ctx.niter} steps')

        # Calculate all the cheap things:
        self.dft.energies()
        self.dft.dipole()
        self.dft.magmoms()

        self.dft.write_converged()

        self.hooks.get('converged', lambda: None)()

        print("\n</div> EXIT New.ASECalculator.iconverge\n")




    def calculate_property(self,
                           atoms: Atoms | None,
                           prop: str) -> Any:
        """Calculate (if not already calculated) a property.

        The ``prop`` string must be one of

        * energy
        * forces
        * stress
        * magmom
        * magmoms
        * dipole
        """

        # ffr: do SCF first?
        for _ in self.iconverge(atoms):
            pass

        if prop == 'forces':
            with self.timer('Forces'):
                self.dft.forces()
        elif prop == 'stress':
            with self.timer('Stress'):
                self.dft.stress()
        elif prop not in self.dft.results:
            raise KeyError('Unknown property:', prop)

        return self.dft.results[prop] * units[prop]

    def get_property(self,
                     name: str,
                     atoms: Atoms | None = None,
                     allow_calculation: bool = True) -> Any:
        if not allow_calculation:
            if name not in self.dft.results:
                return None
            if atoms is None or len(self.check_state(atoms)) == 0:
                return self.dft.results[name] * units[name]
            return None
        if atoms is None:
            atoms = self.atoms
        return self.calculate_property(atoms, name)

    def calculation_required(self, atoms, properties):
        if any(prop not in self.dft.results for prop in properties):
            return True
        return len(self.check_state(atoms)) > 0

    @property
    def results(self):
        if self._dft is None:
            return {}
        return {name: value * units[name]
                for name, value in self.dft.results.items()}

    @trace
    def create_new_calculation(self, atoms: Atoms) -> None:
        with self.timer('Init'):
            self._dft = DFTCalculation.from_parameters(
                atoms, self.params, self.comm, self.log)
        self._atoms = atoms.copy()

    def create_new_calculation_from_old(self, atoms: Atoms) -> None:
        with self.timer('Morph'):
            self._dft = self.dft.new(
                atoms, self.params, self.log)
        self._atoms = atoms.copy()

    def move_atoms(self, atoms):
        with self.timer('Move'):
            self._dft = self.dft.move_atoms(atoms)
        self._atoms = atoms.copy()

    def _calculate_forces(self) -> Array2D:  # units: Ha/Bohr
        """Helper method for force-convergence criterium."""
        with self.timer('Forces'):
            self.dft.forces(silent=True)
        return self.dft.results['forces'].copy()

    def __del__(self):
        self.log('---')
        self.timer.write(self.log)
        try:
            mib = maxrss() / 1024**2
            self.log(f'\nMax RSS: {mib:.3f}  # MiB')
        except NameError:
            pass

    def get_potential_energy(self,
                             atoms: Atoms | None = None,
                             force_consistent: bool = False) -> float:
        return self.calculate_property(atoms,
                                       'free_energy' if force_consistent else
                                       'energy')

    @trace
    def get_forces(self, atoms: Atoms | None = None) -> Array2D:
        return self.calculate_property(atoms, 'forces')

    @trace
    def get_stress(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'stress')

    def get_dipole_moment(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'dipole')

    def get_magnetic_moment(self, atoms: Atoms | None = None) -> float:
        return self.calculate_property(atoms, 'magmom')

    def get_magnetic_moments(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'magmoms')

    def get_non_collinear_magnetic_moment(self,
                                          atoms: Atoms | None = None
                                          ) -> Array1D:
        return self.calculate_property(atoms, 'non_collinear_magmom')

    def get_non_collinear_magnetic_moments(self,
                                           atoms: Atoms | None = None
                                           ) -> Array2D:
        return self.calculate_property(atoms, 'non_collinear_magmoms')

    def check_state(self, atoms, tol=1e-12):
        return list(compare_atoms(self.atoms, atoms))

    def write(self,
              filename: str | Path,
              mode: str = '',
              precision: str = 'double',
              include_projections: bool = True) -> None:
        """Write calculator object to a file.

        Parameters
        ----------
        filename:
            File to be written.
        mode:
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        precision:
            'double' (the default) or 'single'.
        include_projections:
            Use ``include_projections=False`` to not include
            the PAW-projections.
        """
        self.log(f'# Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.dft, skip_wfs=mode != 'all', precision=precision,
                  include_projections=include_projections)

    # Old API:

    implemented_properties = ['energy', 'free_energy',
                              'forces', 'stress',
                              'dipole', 'magmom', 'magmoms']

    def icalculate(self, atoms, system_changes=None):
        yield from self.iconverge(atoms)

    def new(self, **kwargs) -> ASECalculator:
        kwargs = {**dict(self.params.items()), **kwargs}
        return GPAW(**kwargs)

    def get_pseudo_wave_function(self, band, kpt=0, spin=None,
                                 periodic=False,
                                 broadcast=True,
                                 pad=True) -> Array3D | None:
        psit_R = self.dft.wave_functions(n1=band, n2=band + 1,
                                         kpt=kpt, spin=spin,
                                         periodic=periodic,
                                         broadcast=broadcast,
                                         _pad=pad)[0]
        if psit_R is not None:
            return psit_R.data
        return None

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def get_fermi_level(self) -> float:
        return self.dft.ibzwfs.fermi_level * Ha

    def get_fermi_levels(self) -> Array1D:
        fl = self.dft.ibzwfs.fermi_levels
        assert fl is not None
        if len(fl) == 1:
            raise ValueError('Only one Fermi-level.')
        return fl * Ha

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        return self.dft.ibzwfs.get_homo_lumo(spin) * Ha

    def get_number_of_electrons(self):
        return self.dft.ibzwfs.nelectrons

    def get_number_of_bands(self):
        return self.dft.ibzwfs.nbands

    def get_number_of_grid_points(self):
        return self.dft.density.nt_sR.desc.size

    def get_effective_potential(self, spin=0, broadcast=True):
        assert spin == 0
        vt_R = self.dft.potential.vt_sR[spin]
        vt_R = vt_R.to_pbc_grid().gather(broadcast=broadcast)
        return None if vt_R is None else vt_R.data * Ha

    def get_electrostatic_potential(self):
        density = self.dft.density
        potential, _ = self.dft.pot_calc.calculate(density)
        vHt_x = potential.vHt_x
        if isinstance(vHt_x, UGArray):
            return vHt_x.gather(broadcast=True).to_pbc_grid().data * Ha

        grid = self.dft.pot_calc.fine_grid
        return vHt_x.ifft(grid=grid).gather(broadcast=True).data * Ha

    def get_atomic_electrostatic_potentials(self):
        return self.dft.electrostatic_potential().atomic_potentials()

    def get_electrostatic_corrections(self):
        return self.dft.electrostatic_potential().atomic_corrections()

    def get_pseudo_density(self,
                           spin=None,
                           gridrefinement=1,
                           broadcast=True) -> Array3D:
        assert spin is None
        nt_sr = self.dft.densities().pseudo_densities(
            grid_refinement=gridrefinement)
        nt_sr = nt_sr.gather(broadcast=broadcast)
        return None if nt_sr is None else nt_sr.data.sum(0)

    def get_all_electron_density(self,
                                 spin=None,
                                 gridrefinement=1,
                                 broadcast=True,
                                 skip_core=False):
        n_sr = self.dft.densities().all_electron_densities(
            grid_refinement=gridrefinement,
            skip_core=skip_core)
        if spin is None:
            n_sr = n_sr.gather(broadcast=broadcast)
            return None if n_sr is None else n_sr.data.sum(0)
        n_r = n_sr[spin].gather(broadcast=broadcast)
        return None if n_sr is None else n_r.data

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        eig_n = self.dft.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[0] * Ha
        if broadcast:
            if self.comm.rank != 0:
                eig_n = np.empty(self.dft.ibzwfs.nbands)
            self.comm.broadcast(eig_n, 0)
        return eig_n

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True,
                               raw=False):
        ibzwfs = self.dft.ibzwfs
        occ_n = ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[1]
        if not raw:
            weight = ibzwfs.ibz.weight_k[kpt] * ibzwfs.spin_degeneracy
            occ_n *= weight
        if broadcast:
            if self.comm.rank != 0:
                occ_n = np.empty(ibzwfs.nbands)
            self.comm.broadcast(occ_n, 0)
        return occ_n

    def get_reference_energy(self):
        return self.dft.setups.Eref * Ha

    def get_number_of_iterations(self):
        return self.dft.scf_loop.niter

    def get_bz_k_points(self):
        return self.dft.ibzwfs.ibz.bz.kpt_Kc.copy()

    def get_ibz_k_points(self):
        return self.dft.ibzwfs.ibz.kpt_kc.copy()

    def get_k_point_weights(self):
        return self.dft.ibzwfs.ibz.weight_k

    def get_orbital_magnetic_moments(self):
        """Return the orbital magnetic moment vector for each atom."""
        density = self.dft.density
        if density.collinear:
            raise CalculationModeError(
                'Calculator is in collinear mode. '
                'Collinear calculations require spin–orbit '
                'coupling for nonzero orbital magnetic moments.')
        if not self.params.soc:
            warnings.warn('Non-collinear calculation was performed '
                          'without spin–orbit coupling. Orbital '
                          'magnetic moments may not be accurate.')
        return density.calculate_orbital_magnetic_moments()

    def calculate(self, atoms, properties=None, system_changes=None):
        if properties is None:
            properties = ['energy']

        for name in properties:
            self.calculate_property(atoms, name)

    @cached_property
    def wfs(self):
        from my_gpaw25.new.backwards_compatibility import FakeWFS
        return FakeWFS(self.dft.ibzwfs,
                       self.dft.density,
                       self.dft.potential,
                       self.dft.setups,
                       self.comm,
                       self.dft.scf_loop.occ_calc,
                       self.dft.scf_loop.hamiltonian,
                       self.atoms,
                       scale_pw_coefs=True)

    @property
    def density(self):
        from my_gpaw25.new.backwards_compatibility import FakeDensity
        return FakeDensity(self.dft)

    @property
    def hamiltonian(self):
        from my_gpaw25.new.backwards_compatibility import FakeHamiltonian
        return FakeHamiltonian(
            self.dft.ibzwfs, self.dft.density, self.dft.potential,
            self.dft.pot_calc, self.dft.results.get('free_energy'))

    @property
    def spos_ac(self):
        return self.atoms.get_scaled_positions()

    @property
    def world(self):
        return self.comm

    @property
    def setups(self):
        return self.dft.setups

    @property
    def initialized(self):
        return self._dft is not None

    def get_xc_functional(self):
        return self.dft.pot_calc.xc.name

    def get_xc_difference(self, xcparams):
        """Calculate non-selfconsistent XC-energy difference."""
        dft = self.dft
        pot_calc = dft.pot_calc
        density = dft.density
        xc = create_functional(xcparams, pot_calc.fine_grid)
        if xc.type == 'MGGA' and density.taut_sR is None:
            dft.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
            if isinstance(dft.ibzwfs.wfs_qs[0][0].psit_nX, SimpleNamespace):
                params = InputParameters(dict(self.params.items()))
                builder = create_builder(self.atoms, params, self.comm)
                basis_set = builder.create_basis_set()
                ibzwfs = builder.create_ibz_wave_functions(
                    basis_set, dft.potential, log=dft.log)
                ibzwfs.fermi_levels = dft.ibzwfs.fermi_levels
                dft.ibzwfs = ibzwfs
                dft.scf_loop.update_density_and_potential = False
                dft.converge()
            density.update_ked(dft.ibzwfs)
        exct = pot_calc.calculate_non_selfconsistent_exc(xc, density)
        dexc = 0.0
        for a, D_sii in density.D_asii.items():
            setup = self.setups[a]
            dexc += xc.calculate_paw_correction(
                setup, np.array([pack_density(D_ii) for D_ii in D_sii.real]))
        dexc = dft.ibzwfs.domain_comm.sum_scalar(dexc)
        return (exct + dexc - dft.potential.energies['xc']) * Ha

    def diagonalize_full_hamiltonian(self,
                                     nbands: int | None = None,
                                     scalapack=None,
                                     expert: bool | None = None) -> None:
        if expert is not None:
            warnings.warn('Ignoring deprecated "expert" argument',
                          DeprecationWarning)
        dft = self.dft

        if nbands is None:
            nbands = min(wfs.array_shape(global_shape=True)[0]
                         for wfs in dft.ibzwfs)
            nbands = dft.ibzwfs.kpt_comm.min_scalar(nbands)
            assert isinstance(nbands, int)

        dft.scf_loop.occ_calc._set_nbands(nbands)
        ibzwfs = diagonalize(dft.potential,
                             dft.ibzwfs,
                             dft.scf_loop.occ_calc,
                             nbands)
        dft.ibzwfs = ibzwfs
        self.params._add('nbands', ibzwfs.nbands)

    def gs_adapter(self):
        from my_gpaw25.response.groundstate import ResponseGroundStateAdapter
        return ResponseGroundStateAdapter(self)

    def fixed_density(self,
                      *,
                      txt='-',
                      update_fermi_level: bool = False,
                      **kwargs):
        kwargs = {**dict(self.params.items()), **kwargs}

        params = InputParameters(kwargs)
        log = Logger(txt, self.comm)
        builder = create_builder(self.atoms, params, self.comm)
        basis_set = builder.create_basis_set()
        dft = self.dft
        comm1 = dft.ibzwfs.kpt_band_comm
        comm2 = builder.communicators['D']
        potential = dft.potential.redist(
            builder.grid,
            builder.electrostatic_potential_desc,
            builder.atomdist,
            comm1, comm2)
        density = dft.density.redist(builder.grid,
                                     builder.interpolation_desc,
                                     builder.atomdist,
                                     comm1, comm2)
        ibzwfs = builder.create_ibz_wave_functions(basis_set, potential,
                                                   log=log)
        ibzwfs.fermi_levels = dft.ibzwfs.fermi_levels

        scf_loop = builder.create_scf_loop()
        scf_loop.update_density_and_potential = False
        scf_loop.fix_fermi_level = not update_fermi_level

        dft = DFTCalculation(
            ibzwfs, density, potential,
            builder.setups,
            scf_loop,
            SimpleNamespace(relpos_ac=self.dft.relpos_ac,
                            poisson_solver=None,
                            xc=self.dft.pot_calc.xc),
            log)

        dft.converge()

        return ASECalculator(params,
                             log=log,
                             dft=dft,
                             atoms=self.atoms)

    def initialize(self, atoms):
        self.create_new_calculation(atoms)

    def converge_wave_functions(self):
        self.dft.ibzwfs.make_sure_wfs_are_read_from_gpw_file()

    def get_number_of_spins(self):
        return self.dft.density.ndensities

    @property
    def parameters(self):
        return self.params

    def dos(self,
            soc: bool = False,
            theta: float = 0.0,  # degrees
            phi: float = 0.0,  # degrees
            shift_fermi_level: bool = True) -> DOSCalculator:
        """Create DOS-calculator.

        Default is to ``shift_fermi_level`` to 0.0 eV.  For ``soc=True``,
        angles can be given in degrees.
        """
        return DOSCalculator.from_calculator(
            self, soc=soc,
            theta=theta, phi=phi,
            shift_fermi_level=shift_fermi_level)

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure
        return get_band_structure(calc=self)

    @property
    def symmetry(self):
        return self.dft.ibzwfs.ibz.symmetries._old_symmetry

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""
        from my_gpaw25.new.wannier import get_wannier_integrals
        grid = self.dft.density.nt_sR.desc
        k_kc = self.dft.ibzwfs.ibz.bz.kpt_Kc
        G_c = k_kc[nextkpoint] - k_kc[kpoint] - G_I

        return get_wannier_integrals(self.dft.ibzwfs,
                                     grid,
                                     spin, kpoint, nextkpoint, G_c, nbands)

    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin, nbands):
        from my_gpaw25.new.wannier import initial_wannier
        return initial_wannier(self.dft.ibzwfs,
                               initialwannier, kpointgrid, fixedstates,
                               edf, spin, nbands)

    def initialize_positions(self, atoms=None):
        pass

    def set(self, eigensolver):
        from my_gpaw25.new.pwfd.etdm import ETDMPWFD
        self.dft.scf_loop.eigensolver = ETDMPWFD(self.setups,
                                                 self.comm,
                                                 self.atoms,
                                                 eigensolver)

    def todict(self):
        return dict(self.params.items())

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        from my_gpaw25.xc.bee import BEEFEnsemble
        if type not in ['beefvdw', 'mbeef', 'mbeefvdw']:
            raise NotImplementedError('Not implemented for type = %s' % type)
        # assert self.scf.converged
        bee = BEEFEnsemble(self)
        x = bee.create_xc_contributions('exch')
        c = bee.create_xc_contributions('corr')
        if type == 'beefvdw':
            return np.append(x, c)
        elif type == 'mbeef':
            return x.flatten()
        elif type == 'mbeefvdw':
            return np.append(x.flatten(), c)
