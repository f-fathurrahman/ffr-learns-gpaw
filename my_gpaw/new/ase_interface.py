from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, Union

import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha
from my_gpaw import __version__
from my_gpaw.core.uniform_grid import UniformGridFunctions
from my_gpaw.dos import DOSCalculator
from my_gpaw.new import Timer, cached_property
from my_gpaw.new.builder import builder as create_builder
from my_gpaw.new.calculation import (DFTCalculation, DFTState,
                                  ReuseWaveFunctionsError, units)
from my_gpaw.new.gpw import read_gpw, write_gpw
from my_gpaw.new.input_parameters import InputParameters
from my_gpaw.new.logger import Logger
from my_gpaw.new.pw.fulldiag import diagonalize
from my_gpaw.new.xc import XCFunctional
from my_gpaw.typing import Array1D, Array2D, Array3D
from my_gpaw.utilities import pack
from my_gpaw.utilities.memory import maxrss


def GPAW(filename: Union[str, Path, IO[str]] = None,
         **kwargs) -> ASECalculator:
    """Create ASE-compatible GPAW calculator."""
    params = InputParameters(kwargs)
    txt = params.txt
    if txt == '?':
        txt = '-' if filename is None else None
    world = params.parallel['world']
    log = Logger(txt, world)

    if filename is not None:
        assert set(kwargs) <= {'txt', 'parallel', 'communicator'}, kwargs
        atoms, calculation, params, _ = read_gpw(filename, log,
                                                 params.parallel)
        return ASECalculator(params, log, calculation, atoms)

    write_header(log, world, params)
    return ASECalculator(params, log)


def write_header(log, world, params):
    from my_gpaw.io.logger import write_header as header
    log(f'#  __  _  _\n# | _ |_)|_||  |\n# |__||  | ||/\\| - {__version__}\n')
    header(log, world)
    log('---')
    with log.indent('input parameters:'):
        log(**{k: v for k, v in params.items()})


def compare_atoms(a1: Atoms, a2: Atoms) -> set[str]:
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

    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 calculation=None,
                 atoms=None):
        self.params = params
        self.log = log
        self.calculation = calculation

        self.atoms = atoms
        self.timer = Timer()

    def __repr__(self):
        params = []
        for key, value in self.params.items():
            val = repr(value)
            if len(val) > 40:
                val = '...'
            params.append((key, val))
        p = ', '.join(f'{key}: {val}' for key, val in params)
        return f'ASECalculator({p})'

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        """Calculate (if not already calculated) a property.

        The ``prop`` string must be one of

        * energy
        * forces
        * stress
        * magmom
        * magmoms
        * dipole
        """
        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'numbers', 'pbc', 'cell'}:
                if 'numbers' not in changes:
                    # Remember magmoms if there are any:
                    magmom_a = self.calculation.results.get('magmoms')
                    if magmom_a is not None and magmom_a.any():
                        atoms = atoms.copy()
                        atoms.set_initial_magnetic_moments(magmom_a)

                if changes & {'numbers', 'pbc'}:
                    # Start from scratch:
                    self.calculation = None
                else:
                    ibzwfs = self.calculation.state.ibzwfs
                    kpt_parallel_only = (ibzwfs.band_comm.size == 1 and
                                         ibzwfs.domain_comm.size == 1)
                    if kpt_parallel_only:
                        try:
                            self.create_new_calculation_from_old(atoms)
                        except ReuseWaveFunctionsError:
                            self.calculation = None
                        else:
                            self.converge()
                            changes = set()
                    else:
                        # Not implemented: just start from scratch
                        self.calculation = None

        if self.calculation is None:
            self.create_new_calculation(atoms)
            assert self.calculation is not None
            self.converge()
        elif changes:
            self.move_atoms(atoms)
            self.converge()

        if prop not in self.calculation.results:
            if prop == 'forces':
                with self.timer('Forces'):
                    self.calculation.forces()
            elif prop == 'stress':
                with self.timer('Stress'):
                    self.calculation.stress()
            elif prop == 'dipole':
                self.calculation.dipole()
            else:
                raise KeyError('Unknown property:', prop)

        return self.calculation.results[prop] * units[prop]

    def get_property(self,
                     name: str,
                     atoms: Atoms | None = None,
                     allow_calculation: bool = True) -> Any:
        if not allow_calculation and name not in self.calculation.results:
            return None
        if atoms is None:
            atoms = self.atoms
        return self.calculate_property(atoms, name)

    @property
    def results(self):
        if self.calculation is None:
            return {}
        return {name: value * units[name]
                for name, value in self.calculation.results.items()}

    def create_new_calculation(self, atoms: Atoms) -> None:
        with self.timer('Init'):
            self.calculation = DFTCalculation.from_parameters(
                atoms, self.params, self.log)
        self.atoms = atoms.copy()

    def create_new_calculation_from_old(self, atoms: Atoms) -> None:
        with self.timer('Morph'):
            self.calculation = self.calculation.new(
                atoms, self.params, self.log)
        self.atoms = atoms.copy()

    def move_atoms(self, atoms):
        with self.timer('Move'):
            self.calculation = self.calculation.move_atoms(atoms)
        self.atoms = atoms.copy()

    def converge(self):
        """Iterate to self-consistent solution.

        Will also calculate "cheap" properties: energy, magnetic moments
        and dipole moment.
        """
        with self.timer('SCF'):
            self.calculation.converge(calculate_forces=self._calculate_forces)

        # Calculate all the cheap things:
        self.calculation.energies()
        self.calculation.dipole()
        self.calculation.magmoms()

        self.calculation.write_converged()

    def _calculate_forces(self) -> Array2D:  # units: Ha/Bohr
        """Helper method for force-convergence criterium."""
        with self.timer('Forces'):
            self.calculation.forces(silent=True)
        return self.calculation.results['forces']

    def __del__(self):
        try:
            self.log('---')
            self.timer.write(self.log)
            mib = maxrss() / 1024**2
            self.log(f'\nMax RSS: {mib:.3f}  # MiB')
        except (NameError, AttributeError):
            pass

    def get_potential_energy(self,
                             atoms: Atoms,
                             force_consistent: bool = False) -> float:
        return self.calculate_property(atoms,
                                       'free_energy' if force_consistent else
                                       'energy')

    def get_forces(self, atoms: Atoms) -> Array2D:
        return self.calculate_property(atoms, 'forces')

    def get_stress(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'stress')

    def get_dipole_moment(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'dipole')

    def get_magnetic_moment(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'magmom')

    def get_magnetic_moments(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'magmoms')

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename:
            File to be written
        mode:
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'# Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.calculation, skip_wfs=mode != 'all')

    # Old API:

    implemented_properties = ['energy', 'free_energy',
                              'forces', 'stress',
                              'dipole', 'magmom', 'magmoms']

    def new(self, **kwargs) -> ASECalculator:
        kwargs = {**dict(self.params.items()), **kwargs}
        return GPAW(**kwargs)

    def get_pseudo_wave_function(self, band, kpt=0, spin=0,
                                 periodic=False) -> Array3D:
        state = self.calculation.state
        wfs = state.ibzwfs.get_wfs(spin=spin, kpt=kpt, n1=band, n2=band + 1)
        basis = getattr(self.calculation.scf_loop.hamiltonian, 'basis', None)
        grid = state.density.nt_sR.desc
        wfs = wfs.to_uniform_grid_wave_functions(grid, basis)
        psit_R = wfs.psit_nX[0]
        if not psit_R.desc.pbc.all():
            psit_R = psit_R.to_pbc_grid()
        if periodic:
            psit_R.multiply_by_eikr(-psit_R.desc.kpt_c)
        return psit_R.data * Bohr**-1.5

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def get_fermi_level(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    def get_fermi_levels(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 2
        return fl

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        state = self.calculation.state
        return state.ibzwfs.get_homo_lumo(spin) * Ha

    def get_number_of_electrons(self):
        state = self.calculation.state
        return state.ibzwfs.nelectrons

    def get_number_of_bands(self):
        state = self.calculation.state
        return state.ibzwfs.nbands

    def get_number_of_grid_points(self):
        return self.calculation.state.density.nt_sR.desc.size

    def get_effective_potential(self, spin=0):
        assert spin == 0
        vt_R = self.calculation.state.potential.vt_sR[spin]
        return vt_R.to_pbc_grid().data * Ha

    def get_electrostatic_potential(self):
        density = self.calculation.state.density
        potential, vHt_x, W_aL = self.calculation.pot_calc.calculate(density)
        if isinstance(vHt_x, UniformGridFunctions):
            return vHt_x.to_pbc_grid().data * Ha

        return vHt_x.interpolate(
            grid=self.calculation.pot_calc.fine_grid).data * Ha

    def get_atomic_electrostatic_potentials(self):
        return self.calculation.electrostatic_potential().atomic_potentials()

    def get_electrostatic_corrections(self):
        return self.calculation.electrostatic_potential().atomic_corrections()

    def get_pseudo_density(self, spin=None, gridrefinement=1):
        assert spin is None
        nt_sr = self.calculation.densities().pseudo_densities(
            grid_refinement=gridrefinement)
        return nt_sr.to_pbc_grid().data.sum(0)

    def get_all_electron_density(self,
                                 spin=None,
                                 gridrefinement=1,
                                 skip_core=False):
        assert spin is None
        n_sr = self.calculation.densities().all_electron_densities(
            grid_refinement=gridrefinement,
            skip_core=skip_core)
        return n_sr.to_pbc_grid().data.sum(0)

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        state = self.calculation.state
        eig_n = state.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[0] * Ha
        if broadcast:
            if self.world.rank != 0:
                eig_n = np.empty(state.ibzwfs.nbands)
            self.world.broadcast(eig_n, 0)
        return eig_n

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True):
        state = self.calculation.state
        weight = state.ibzwfs.ibz.weight_k[kpt] * state.ibzwfs.spin_degeneracy
        occ_n = state.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[1] * weight
        if broadcast:
            if self.world.rank != 0:
                occ_n = np.empty(state.ibzwfs.nbands)
            self.world.broadcast(occ_n, 0)
        return occ_n

    def get_reference_energy(self):
        return self.calculation.setups.Eref * Ha

    def get_number_of_iterations(self):
        return self.calculation.scf_loop.niter

    def get_bz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.bz.kpt_Kc.copy()

    def get_ibz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.kpt_kc.copy()

    def calculate(self, atoms, properties=None, system_changes=None):
        if properties is None:
            properties = ['energy']

        for name in properties:
            self.calculate_property(atoms, name)
        # self.get_potential_energy(atoms)

    @cached_property
    def wfs(self):
        from my_gpaw.new.backwards_compatibility import FakeWFS
        return FakeWFS(self.calculation, self.atoms)

    @property
    def density(self):
        from my_gpaw.new.backwards_compatibility import FakeDensity
        return FakeDensity(self.calculation)

    @property
    def hamiltonian(self):
        from my_gpaw.new.backwards_compatibility import FakeHamiltonian
        return FakeHamiltonian(self.calculation)

    @property
    def spos_ac(self):
        return self.atoms.get_scaled_positions()

    @property
    def world(self):
        return self.calculation.scf_loop.world

    @property
    def setups(self):
        return self.calculation.setups

    @property
    def initialized(self):
        return self.calculation is not None

    def get_xc_difference(self, xcparams):
        """Calculate non-selfconsistent XC-energy difference."""
        state = self.calculation.state
        xc = XCFunctional(xcparams, state.density.ncomponents)
        exct = self.calculation.pot_calc.calculate_non_selfconsistent_exc(
            state.density.nt_sR, xc)
        dexc = 0.0
        for a, D_sii in state.density.D_asii.items():
            setup = self.setups[a]
            dexc += xc.calculate_paw_correction(
                setup,
                np.array([pack(D_ii) for D_ii in D_sii]))
        return (exct + dexc - state.potential.energies['xc']) * Ha

    def diagonalize_full_hamiltonian(self,
                                     nbands: int = None,
                                     scalapack=None,
                                     expert: bool = None) -> None:
        if expert is not None:
            warnings.warn('Ignoring deprecated "expert" argument')
        state = self.calculation.state
        ibzwfs = diagonalize(state.potential,
                             state.ibzwfs,
                             self.calculation.scf_loop.occ_calc,
                             nbands)
        self.calculation.state = DFTState(ibzwfs,
                                          state.density,
                                          state.potential)
        nbands = ibzwfs.nbands
        self.params.nbands = nbands
        self.params.keys.append('nbands')

    def gs_adapter(self):
        from my_gpaw.response.groundstate import ResponseGroundStateAdapter
        return ResponseGroundStateAdapter(self)

    def fixed_density(self, **kwargs):
        kwargs = {**dict(self.params.items()), **kwargs}
        params = InputParameters(kwargs)
        txt = params.txt
        if txt == '?':
            txt = '-'
        world = params.parallel['world']
        log = Logger(txt, world)
        builder = create_builder(self.atoms, params)
        basis_set = builder.create_basis_set()
        state = self.calculation.state
        ibzwfs = builder.create_ibz_wave_functions(basis_set, state.potential,
                                                   log=log)
        ibzwfs.fermi_levels = state.ibzwfs.fermi_levels
        state = DFTState(ibzwfs, state.density, state.potential)
        scf_loop = builder.create_scf_loop()
        scf_loop.update_density_and_potential = False

        calculation = DFTCalculation(
            state,
            builder.setups,
            scf_loop,
            SimpleNamespace(fracpos_ac=self.calculation.fracpos_ac,
                            poisson_solver=None),
            log)

        calculation.converge()

        return ASECalculator(params, log, calculation, self.atoms)

    def initialize(self, atoms):
        self.create_new_calculation(atoms)

    def converge_wave_functions(self):
        self.calculation.state.ibzwfs.make_sure_wfs_are_read_from_gpw_file()

    def get_number_of_spins(self):
        return self.calculation.state.density.ndensities

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
