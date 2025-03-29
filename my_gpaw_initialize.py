import numpy as np
import my_gpaw.mpi as mpi
from ase.units import Bohr
from my_gpaw.xc.kernel import XCKernel
from my_gpaw.xc import XC
from my_gpaw.utilities.gpts import get_number_of_grid_points
from my_gpaw.convergence_criteria import dict2criterion
from my_gpaw.output import (print_cell, print_parallelization_details,
                         print_positions)

from my_gpaw_create_wave_functions import my_gpaw_create_wave_functions

def my_gpaw_initialize(calc, atoms=None, reading=False):

    """Inexpensive initialization."""

    calc.log('Initialize ...\n')

    if atoms is None:
        atoms = calc.atoms
    else:
        atoms = atoms.copy()
        calc._set_atoms(atoms)

    par = calc.parameters

    natoms = len(atoms)

    cell_cv = atoms.get_cell() / Bohr
    number_of_lattice_vectors = cell_cv.any(axis=1).sum()
    if number_of_lattice_vectors < 3:
        raise ValueError(
            'GPAW requires 3 lattice vectors.  Your system has {}.'
            .format(number_of_lattice_vectors))

    pbc_c = atoms.get_pbc()
    assert len(pbc_c) == 3
    magmom_a = atoms.get_initial_magnetic_moments()

    if par.experimental.get('magmoms') is not None:
        magmom_av = np.array(par.experimental['magmoms'], float)
        collinear = False
    else:
        magmom_av = np.zeros((natoms, 3))
        magmom_av[:, 2] = magmom_a
        collinear = True

    # Need this?
    mpi.synchronize_atoms(atoms, calc.world)

    # Generate new xc functional only when it is reset by set
    # XXX sounds like this should use the _changed_keywords dictionary.
    if calc.hamiltonian is None or calc.hamiltonian.xc is None:
        if isinstance(par.xc, (str, dict, XCKernel)):
            xc = XC(par.xc, collinear=collinear, atoms=atoms)
        else:
            xc = par.xc
    else:
        xc = calc.hamiltonian.xc

    if par.fixdensity:
        warnings.warn(
            'The fixdensity keyword has been deprecated. '
            'Please use the GPAW.fixed_density() method instead.',
            DeprecationWarning)
        if calc.hamiltonian.xc.type == 'MGGA':
            raise ValueError('MGGA does not support deprecated '
                                'fixdensity option.')

    mode = par.mode
    if isinstance(mode, str):
        mode = {'name': mode}
    if isinstance(mode, dict):
        mode = create_wave_function_mode(**mode)

    #breakpoint()

    if par.dtype == complex:
        warnings.warn('Use mode={}(..., force_complex_dtype=True) '
                        'instead of dtype=complex'.format(mode.name.upper()))
        mode.force_complex_dtype = True
        del par['dtype']
        par.mode = mode

    if xc.orbital_dependent and mode.name == 'lcao':
        raise ValueError('LCAO mode does not support '
                            'orbital-dependent XC functionals.')

    realspace = mode.interpolation != 'fft'

    calc.create_setups(mode, xc)

    #breakpoint()

    if not realspace:
        pbc_c = np.ones(3, bool)

    magnetic = magmom_av.any()

    if par.hund:
        spinpol = True
        magnetic = True
        c = par.charge / natoms
        for a, setup in enumerate(calc.setups):
            magmom_av[a, 2] = setup.get_hunds_rule_moment(c)

    if collinear:
        spinpol = par.spinpol
        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                                'spin-paired calculation!')
        nspins = 1 + int(spinpol)

        if spinpol:
            calc.log('Spin-polarized calculation.')
            calc.log('Magnetic moment: {:.6f}\n'.format(magmom_av.sum()))
        else:
            calc.log('Spin-paired calculation\n')
    else:
        nspins = 1
        calc.log('Non-collinear calculation.')
        calc.log('Magnetic moment: ({:.6f}, {:.6f}, {:.6f})\n'
                    .format(*magmom_av.sum(0)))

    calc.create_symmetry(magmom_av, cell_cv, reading)

    if par.gpts is not None:
        if par.h is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")
        N_c = np.array(par.gpts)
        h = None
    else:
        h = par.h
        if h is not None:
            h /= Bohr
        if h is None and reading:
            shape = calc.reader.density.proxy('density').shape[-3:]
            N_c = 1 - pbc_c + shape
        elif h is None and calc.density is not None:
            N_c = calc.density.gd.N_c
        else:
            N_c = get_number_of_grid_points(cell_cv, h, mode, realspace,
                                            calc.symmetry)

    calc.setups.set_symmetry(calc.symmetry)

    if not collinear and len(calc.symmetry.op_scc) > 1:
        raise ValueError('Can''t use symmetries with non-collinear '
                            'calculations')

    if isinstance(par.background_charge, dict):
        background = create_background_charge(**par.background_charge)
    else:
        background = par.background_charge

    nao = calc.setups.nao
    nvalence = calc.setups.nvalence - par.charge
    if par.background_charge is not None:
        nvalence += background.charge

    M = np.linalg.norm(magmom_av.sum(0))

    nbands = par.nbands

    orbital_free = any(setup.orbital_free for setup in calc.setups)
    if orbital_free:
        nbands = 1

    if isinstance(nbands, str):
        if nbands == 'nao':
            nbands = nao
        elif nbands[-1] == '%':
            basebands = (nvalence + M) / 2
            nbands = int(np.ceil(float(nbands[:-1]) / 100 * basebands))
        else:
            raise ValueError('Integer expected: Only use a string '
                                'if giving a percentage of occupied bands')

    if nbands is None:
        # Number of bound partial waves:
        nbandsmax = sum(setup.get_default_nbands()
                        for setup in calc.setups)
        nbands = int(np.ceil((1.2 * (nvalence + M) / 2))) + 4
        if nbands > nbandsmax:
            nbands = nbandsmax
        if mode.name == 'lcao' and nbands > nao:
            nbands = nao
    elif nbands <= 0:
        nbands = max(1, int(nvalence + M + 0.5) // 2 + (-nbands))

    if nbands > nao and mode.name == 'lcao':
        raise ValueError('Too many bands for LCAO calculation: '
                            '%d bands and only %d atomic orbitals!' %
                            (nbands, nao))

    if nvalence < 0:
        raise ValueError(
            'Charge %f is not possible - not enough valence electrons' %
            par.charge)

    if nvalence > 2 * nbands and not orbital_free:
        raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                            % (nvalence, nbands))

    # Gather convergence criteria for SCF loop.
    criteria = calc.default_parameters['convergence'].copy()  # keep order
    criteria.update(par.convergence)
    custom = criteria.pop('custom', [])
    del criteria['bands']
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                    'which GPAW does not know how to load. This '
                    'criterion is NOT enabled; you may want to manually'
                    ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
                'Please be sure that each calculator is fed a '
                'unique instance of this criterion. '
                'Note that if you save the calculator instance to '
                'a .gpw file you may not be able to re-open it. '
                .format(criterion.name))
        warnings.warn(msg)

    if calc.scf is None:
        calc.create_scf(criteria, mode)

    if not collinear:
        nbands *= 2

    if not calc.wfs:
        my_gpaw_create_wave_functions(
            calc, mode, realspace,
            nspins, collinear, nbands, nao,
            nvalence, calc.setups,
            cell_cv, pbc_c, N_c,
            xc)
    else:
        calc.wfs.set_setups(calc.setups)

    occ = calc.create_occupations(cell_cv, magmom_av[:, 2].sum(),
                                    orbital_free, nvalence)
    calc.wfs.occupations = occ

    if not calc.wfs.eigensolver:
        calc.create_eigensolver(xc, nbands, mode)

    if calc.density is None and not reading:
        assert not par.fixdensity, 'No density to fix!'

    olddens = None
    if (calc.density is not None and
        (calc.density.gd.parsize_c != calc.wfs.gd.parsize_c).any()):
        # Domain decomposition has changed, so we need to
        # reinitialize density and hamiltonian:
        if par.fixdensity:
            olddens = calc.density

        calc.density = None
        calc.hamiltonian = None

    if calc.density is None:
        calc.create_density(realspace, mode, background, h)

    # XXXXXXXXXX if setups change, then setups.core_charge may change.
    # But that parameter was supplied in Density constructor!
    # This surely is a bug!
    calc.density.initialize(calc.setups, calc.timer,
                            magmom_av, par.hund)
    calc.density.set_mixer(par.mixer)
    if calc.density.mixer.driver.name == 'dummy' or par.fixdensity:
        calc.log('No density mixing\n')
    else:
        calc.log(calc.density.mixer, '\n')
    calc.density.fixed = par.fixdensity
    calc.density.log = calc.log

    if olddens is not None:
        calc.density.initialize_from_other_density(olddens,
                                                    calc.wfs.kptband_comm)

    if calc.hamiltonian is None:
        calc.create_hamiltonian(realspace, mode, xc)

    xc.initialize(calc.density, calc.hamiltonian, calc.wfs)

    description = xc.get_description()
    if description is not None:
        calc.log('XC parameters: {}\n'
                    .format('\n  '.join(description.splitlines())))

    if xc.type == 'GLLB' and olddens is not None:
        xc.heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(olddens)

    calc.print_memory_estimate(maxdepth=3)

    print_parallelization_details(calc.wfs, calc.hamiltonian, calc.log)

    calc.log('Number of atoms:', natoms)
    calc.log('Number of atomic orbitals:', calc.wfs.setups.nao)
    calc.log('Number of bands in calculation:', calc.wfs.bd.nbands)
    calc.log('Number of valence electrons:', calc.wfs.nvalence)

    n = par.convergence.get('bands', 'occupied')
    if isinstance(n, int) and n < 0:
        n += calc.wfs.bd.nbands
    calc.log('Bands to converge:', n)

    calc.log(flush=True)

    calc.timer.print_info(calc)

    #if my_gpaw.dry_run:
    #    calc.dry_run()

    if (realspace and
        calc.hamiltonian.poisson.get_description() == 'FDTD+TDDFT'):
        calc.hamiltonian.poisson.set_density(calc.density)
        calc.hamiltonian.poisson.print_messages(calc.log)
        calc.log.fd.flush()

    calc.initialized = True
    calc.log('... initialized\n')

    print()
    print("     ----------------")
    print("     EXIT: initialize")
    print("     ----------------")
    print()
