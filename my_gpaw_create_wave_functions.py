import my_gpaw.mpi as mpi
from my_gpaw.hybrids import HybridXC
from my_gpaw.band_descriptor import BandDescriptor
from my_gpaw.kohnsham_layouts import get_KohnSham_layouts

def my_gpaw_create_wave_functions(
    calc, mode, realspace,
    nspins, collinear, nbands, nao, nvalence,
    setups, cell_cv, pbc_c, N_c, xc):
    
    print("\n<div> ENTER my_gpaw_create_wave_functions\n")

    par = calc.parameters

    kd = calc.create_kpoint_descriptor(nspins)

    parallelization = mpi.Parallelization(calc.world,
                                            kd.nibzkpts)

    parsize_kpt = calc.parallel['kpt']
    parsize_domain = calc.parallel['domain']
    parsize_bands = calc.parallel['band']

    if isinstance(xc, HybridXC):
        parsize_kpt = 1
        parsize_domain = calc.world.size
        parsize_bands = 1

    ndomains = None
    if parsize_domain is not None:
        ndomains = np.prod(parsize_domain)
    parallelization.set(kpt=parsize_kpt,
                        domain=ndomains,
                        band=parsize_bands)
    comms = parallelization.build_communicators()
    domain_comm = comms['d']
    kpt_comm = comms['k']
    band_comm = comms['b']
    kptband_comm = comms['D']
    domainband_comm = comms['K']

    calc.comms = comms

    kd.set_communicator(kpt_comm)

    parstride_bands = calc.parallel['stridebands']
    if parstride_bands:
        raise RuntimeError('stridebands is unreliable')

    bd = BandDescriptor(nbands, band_comm, parstride_bands)

    # Construct grid descriptor for coarse grids for wave functions:
    gd = calc.create_grid_descriptor(N_c, cell_cv, pbc_c,
                                        domain_comm, parsize_domain)

    if hasattr(calc, 'time') or mode.force_complex_dtype or not collinear:
        dtype = complex
    else:
        if kd.gamma:
            dtype = float
        else:
            dtype = complex

    wfs_kwargs = dict(gd=gd, nvalence=nvalence, setups=setups,
                        bd=bd, dtype=dtype, world=calc.world, kd=kd,
                        kptband_comm=kptband_comm, timer=calc.timer)

    if calc.parallel['sl_auto'] and compiled_with_sl():
        # Choose scalapack parallelization automatically

        for key, val in calc.parallel.items():
            if (key.startswith('sl_') and key != 'sl_auto' and
                val is not None):
                raise ValueError("Cannot use 'sl_auto' together "
                                    "with '%s'" % key)

        max_scalapack_cpus = bd.comm.size * gd.comm.size
        sl_default = suggest_blocking(nbands, max_scalapack_cpus)
    else:
        sl_default = calc.parallel['sl_default']

    if mode.name == 'lcao':
        assert collinear
        # Layouts used for general diagonalizer
        sl_lcao = calc.parallel['sl_lcao']
        if sl_lcao is None:
            sl_lcao = sl_default

        elpasolver = None
        if calc.parallel['use_elpa']:
            elpasolver = calc.parallel['elpasolver']
        lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                        gd, bd, domainband_comm, dtype,
                                        nao=nao, timer=calc.timer,
                                        elpasolver=elpasolver)

        calc.wfs = mode(lcaoksl, **wfs_kwargs)

    elif mode.name == 'fd' or mode.name == 'pw':
        # Use (at most) all available LCAO for initialization
        lcaonbands = min(nbands, nao)

        try:
            lcaobd = BandDescriptor(lcaonbands, band_comm,
                                    parstride_bands)
        except RuntimeError:
            initksl = None
        else:
            # Layouts used for general diagonalizer
            # (LCAO initialization)
            sl_lcao = calc.parallel['sl_lcao']
            if sl_lcao is None:
                sl_lcao = sl_default
            initksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                            gd, lcaobd, domainband_comm,
                                            dtype, nao=nao,
                                            timer=calc.timer)

        reuse_wfs_method = par.experimental.get('reuse_wfs_method', 'paw')
        sl = (domainband_comm,) + (calc.parallel['sl_diagonalize'] or
                                    sl_default or
                                    (1, 1, None))
        calc.wfs = mode(sl, initksl,
                        reuse_wfs_method=reuse_wfs_method,
                        collinear=collinear,
                        **wfs_kwargs)
        print()
        print("!!! Pass here 1460 in ", __file__)
        print()
    else:
        calc.wfs = mode(calc, collinear=collinear, **wfs_kwargs)

    calc.log(calc.wfs, '\n')

    print("\n</div> EXIT my_gpaw_create_wave_functions\n")