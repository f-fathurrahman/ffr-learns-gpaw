from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

import ase.io.ulm as ulm
import my_gpaw
import my_gpaw.mpi as mpi
import numpy as np
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
from my_gpaw.core.atom_arrays import AtomArraysLayout
from my_gpaw.new.builder import builder as create_builder
from my_gpaw.new.calculation import DFTCalculation, DFTState, units
from my_gpaw.new.density import Density
from my_gpaw.new.input_parameters import InputParameters
from my_gpaw.new.logger import Logger
from my_gpaw.new.potential import Potential
from my_gpaw.utilities import unpack, unpack2
from my_gpaw.typing import DTypeLike

ENERGY_NAMES = ['kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
                'total_free', 'total_extrapolated',
                'band']


def write_gpw(filename: str,
              atoms,
              params,
              calculation: DFTCalculation,
              skip_wfs: bool = True) -> None:

    world = params.parallel['world']

    writer: ulm.Writer | ulm.DummyWriter
    if world.rank == 0:
        writer = ulm.Writer(filename, tag='gpaw')
    else:
        writer = ulm.DummyWriter()

    with writer:
        writer.write(version=4,
                     gpaw_version=gpaw.__version__,
                     ha=Ha,
                     bohr=Bohr)

        write_atoms(writer.child('atoms'), atoms)

        # Note that 'non_collinear_magmoms' is not an ASE standard name!
        results = {key: value * units[key]
                   for key, value in calculation.results.items()
                   if key != 'non_collinear_magmoms'}
        writer.child('results').write(**results)

        p = {k: v for k, v in params.items() if k not in ['txt', 'parallel']}
        # ULM does not know about numpy dtypes:
        if 'dtype' in p:
            p['dtype'] = np.dtype(p['dtype']).name
        writer.child('parameters').write(**p)

        state = calculation.state
        state.density.write(writer.child('density'))
        state.potential._write_gpw(writer.child('hamiltonian'),
                                   calculation.state.ibzwfs)
        wf_writer = writer.child('wave_functions')
        state.ibzwfs.write(wf_writer, skip_wfs)

        if not skip_wfs and params.mode['name'] == 'pw':
            write_wave_function_indices(wf_writer,
                                        state.ibzwfs,
                                        state.density.nt_sR.desc)

    world.barrier()


def write_wave_function_indices(writer, ibzwfs, grid):
    if ibzwfs.band_comm.rank != 0:
        return
    if ibzwfs.domain_comm.rank != 0:
        return

    kpt_comm = ibzwfs.kpt_comm
    ibz = ibzwfs.ibz
    nG = ibzwfs.get_max_shape(global_shape=True)[-1]

    writer.add_array('indices', (len(ibz), nG), np.int32)

    index_G = np.zeros(nG, np.int32)
    size = tuple(grid.size)
    if ibzwfs.dtype == float:
        size = (size[0], size[1], size[2] // 2 + 1)

    for k, rank in enumerate(ibzwfs.rank_k):
        if rank == kpt_comm.rank:
            wfs = ibzwfs.wfs_qs[ibzwfs.q_k[k]][0]
            i_G = wfs.psit_nX.desc.indices(size)
            index_G[:len(i_G)] = i_G
            index_G[len(i_G):] = -1
            if rank == 0:
                writer.fill(index_G)
            else:
                kpt_comm.send(index_G, 0)
        elif kpt_comm.rank == 0:
            kpt_comm.receive(index_G, rank)
            writer.fill(index_G)


def read_gpw(filename: Union[str, Path, IO[str]],
             log: Union[Logger, str, Path, IO[str]] = None,
             parallel: dict[str, Any] = None,
             dtype: DTypeLike = None):
    """
    Read gpw file

    Returns
    -------
    atoms, calculation, params, builder
    """
    parallel = parallel or {}
    world = parallel.get('world', mpi.world)

    if not isinstance(log, Logger):
        log = Logger(log, world)

    log(f'Reading from {filename}')

    reader = ulm.Reader(filename)
    bohr = reader.bohr
    ha = reader.ha

    atoms = read_atoms(reader.atoms)
    kwargs = reader.parameters.asdict()
    kwargs['parallel'] = parallel

    if 'dtype' in kwargs:
        kwargs['dtype'] = np.dtype(kwargs['dtype'])

    # kwargs['nbands'] = reader.wave_functions.eigenvalues.shape[-1]

    params = InputParameters(kwargs, warn=False)
    builder = create_builder(atoms, params)

    if world.rank == 0:
        nt_sR_array = reader.density.density * bohr**3
        vt_sR_array = reader.hamiltonian.potential / ha
        D_sap_array = reader.density.atomic_density_matrices
        dH_sap_array = reader.hamiltonian.atomic_hamiltonian_matrices / ha
        shape = nt_sR_array.shape[1:]
    else:
        nt_sR_array = None
        vt_sR_array = None
        D_sap_array = None
        dH_sap_array = None
        shape = None

    if builder.grid.global_shape() != mpi.broadcast(shape, comm=world):
        # old gpw-file:
        kwargs.pop('h', None)
        kwargs['gpts'] = nt_sR_array.shape[1:]
        params = InputParameters(kwargs, warn=False)
        builder = create_builder(atoms, params)

    if dtype is not None:
        params.mode['dtype'] = dtype

    (kpt_comm, band_comm, domain_comm, kpt_band_comm) = (
        builder.communicators[x] for x in 'kbdD')

    nt_sR = builder.grid.empty(builder.ncomponents)
    vt_sR = builder.grid.empty(builder.ncomponents)

    atom_array_layout = AtomArraysLayout([(setup.ni * (setup.ni + 1) // 2)
                                          for setup in builder.setups],
                                         atomdist=builder.atomdist)
    D_asp = atom_array_layout.empty(builder.ncomponents)
    dtype = float if builder.ncomponents < 4 else complex
    dH_asp = atom_array_layout.new(dtype=dtype).empty(builder.ncomponents)

    if kpt_band_comm.rank == 0:
        nt_sR.scatter_from(nt_sR_array)
        vt_sR.scatter_from(vt_sR_array)
        D_asp.scatter_from(D_sap_array)
        dH_asp.scatter_from(dH_sap_array)

    if reader.version < 4:
        convert_to_new_packing_convention(D_asp, density=True)
        convert_to_new_packing_convention(dH_asp)

    kpt_band_comm.broadcast(nt_sR.data, 0)
    kpt_band_comm.broadcast(vt_sR.data, 0)
    kpt_band_comm.broadcast(D_asp.data, 0)
    kpt_band_comm.broadcast(dH_asp.data, 0)

    density = Density.from_data_and_setups(nt_sR, D_asp.to_full(),
                                           builder.params.charge,
                                           builder.setups)
    energies = {name: reader.hamiltonian.get(f'e_{name}', np.nan) / ha
                for name in ENERGY_NAMES}
    penergies = {key: e for key, e in energies.items()
                 if not key.startswith('total')}
    e_band = penergies.pop('band', np.nan)
    e_entropy = penergies.pop('entropy')
    penergies['kinetic'] -= e_band

    potential = Potential(vt_sR, dH_asp.to_full(), penergies)

    ibzwfs = builder.read_ibz_wave_functions(reader)
    ibzwfs.energies = {
        'band': e_band,
        'entropy': e_entropy,
        'extrapolation': (energies['total_extrapolated'] -
                          energies['total_free'])}

    calculation = DFTCalculation(
        DFTState(ibzwfs, density, potential),
        builder.setups,
        builder.create_scf_loop(),
        pot_calc=builder.create_potential_calculator(),
        log=log)

    results = {key: value / units[key]
               for key, value in reader.results.asdict().items()}

    if results:
        log(f'Read {", ".join(sorted(results))}')

    calculation.results = results

    if builder.mode in ['pw', 'fd']:
        data = ibzwfs.wfs_qs[0][0].psit_nX.data
        if not hasattr(data, 'fd'):
            reader.close()
    else:
        reader.close()

    return atoms, calculation, params, builder


def convert_to_new_packing_convention(a_asp, density=False):
    """Convert from old to new convention.

    ::

        1 2 3      1 2 4
        . 4 5  ->  . 3 5
        . . 6      . . 6
    """
    for a_sp in a_asp.values():
        if density:
            a_sii = unpack2(a_sp)
        else:
            a_sii = unpack(a_sp)
        L = np.tril_indices(a_sii.shape[1])
        a_sp[:] = a_sii[(...,) + L]
