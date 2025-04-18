from gpaw.utilities.progressbar import ProgressBar
from gpaw.pw.descriptor import count_reciprocal_vectors
from gpaw.mpi import broadcast_exception

from my_g0w0 import Sigma, QSymmetryOp

from contextlib import ExitStack
from ase.parallel import broadcast


def g0w0_calculate(g0w0calc, qpoints=None):

    print("**** ENTER G0W0Calculator.calculate")

    qpoints = set(qpoints) if qpoints else None
    print("qpoints = ", qpoints)

    if qpoints is None:
        g0w0calc.context.print('Summing all q:')
    else:
        qpt_str = ' '.join(map(str, qpoints))
        g0w0calc.context.print(f'Calculating following q-points: {qpt_str}')
    
    # call main loop
    g0w0_calculate_q_points(g0w0calc, qpoints=qpoints)
    
    if qpoints is not None:
        return f'A partial result of q-points: {qpt_str}'
    
    sigmas = g0w0calc.read_sigmas()
    g0w0calc.all_results = g0w0calc.postprocess(sigmas)
    # Note: g0w0calc.results is a pointer pointing to one of the results,
    # for historical reasons.

    g0w0calc.savepckl()

    print("**** EXIT G0W0Calculator.calculate")

    return g0w0calc.results



def g0w0_calculate_q_points(g0w0calc, qpoints):
    """Main loop over irreducible Brillouin zone points.
    Handles restarts of individual qpoints using FileCache from ASE,
    and subsequently calls calculate_q."""

    pb = ProgressBar(g0w0calc.context.fd)

    g0w0calc.context.timer.start('W')
    g0w0calc.context.print('\nCalculating screened Coulomb potential')
    g0w0calc.context.print(g0w0calc.wcalc.coulomb.description())

    chi0calc = g0w0calc.chi0calc
    g0w0calc.context.print(g0w0calc.wd)

    # Find maximum size of chi-0 matrices:
    nGmax = max(count_reciprocal_vectors(chi0calc.chi0_body_calc.ecut,
                                         g0w0calc.wcalc.gs.gd, q_c)
                for q_c in g0w0calc.wcalc.qd.ibzk_kc)
    nw = len(g0w0calc.wd)

    size = g0w0calc.chi0calc.chi0_body_calc.integrator.blockcomm.size

    mynGmax = (nGmax + size - 1) // size
    mynw = (nw + size - 1) // size

    # some memory sizes...
    if g0w0calc.context.comm.rank == 0:
        siz = (nw * mynGmax * nGmax +
               max(mynw * nGmax, nw * mynGmax) * nGmax) * 16
        sizA = (nw * nGmax * nGmax + nw * nGmax * nGmax) * 16
        g0w0calc.context.print(
            '  memory estimate for chi0: local=%.2f MB, global=%.2f MB'
            % (siz / 1024**2, sizA / 1024**2))

    if g0w0calc.context.comm.rank == 0 and qpoints is None:
        g0w0calc.context.print('Removing empty qpoint cache files...')
        g0w0calc.qcache.strip_empties()

    g0w0calc.context.comm.barrier()

    # Need to pause the timer in between iterations
    g0w0calc.context.timer.stop('W')

    with broadcast_exception(g0w0calc.context.comm):
        if g0w0calc.context.comm.rank == 0:
            for key, sigmas in g0w0calc.qcache.items():
                if qpoints and int(key) not in qpoints:
                    continue
                sigmas = {fxc_mode: Sigma.fromdict(sigma)
                          for fxc_mode, sigma in sigmas.items()}
                for fxc_mode, sigma in sigmas.items():
                    sigma.validate_inputs(g0w0calc.get_validation_inputs())

    for iq, q_c in enumerate(g0w0calc.wcalc.qd.ibzk_kc):
        # If a list of q-points is specified,
        # skip the q-points not in the list
        if qpoints and (iq not in qpoints):
            continue
        with ExitStack() as stack:
            if g0w0calc.context.comm.rank == 0:
                qhandle = stack.enter_context(g0w0calc.qcache.lock(str(iq)))
                skip = qhandle is None
            else:
                skip = False

            skip = broadcast(skip, comm=g0w0calc.context.comm)

            if skip:
                continue

            result = g0w0_calculate_q_point(g0w0calc, iq, q_c, pb, chi0calc)

            if g0w0calc.context.comm.rank == 0:
                qhandle.save(result)
    pb.finish()



def g0w0_calculate_q_point(g0w0calc, iq, q_c, pb, chi0calc):
    # Reset calculation
    sigmashape = (len(g0w0calc.ecut_e), *g0w0calc.shape)
    sigmas = {fxc_mode: Sigma(iq, q_c, fxc_mode, sigmashape,
              len(g0w0calc.evaluate_sigma),
              **g0w0calc.get_validation_inputs())
              for fxc_mode in g0w0calc.fxc_modes}

    chi0 = chi0calc.create_chi0(q_c)

    m1 = chi0calc.gs.nocc1
    for ie, ecut in enumerate(g0w0calc.ecut_e):
        g0w0calc.context.timer.start('W')

        # First time calculation
        if ecut == chi0.qpd.ecut:
            # Nothing to cut away:
            m2 = g0w0calc.nbands
        else:
            m2 = int(g0w0calc.wcalc.gs.volume * ecut**1.5 * 2**0.5 / 3 / pi**2)
            if m2 > g0w0calc.nbands:
                raise ValueError(f'Trying to extrapolate ecut to'
                                 f'larger number of bands ({m2})'
                                 f' than there are bands '
                                 f'({g0w0calc.nbands}).')
        qpdi, Wdict, blocks1d, pawcorr = g0w0calc.calculate_w(
            chi0calc, q_c, chi0,
            m1, m2, ecut, iq)
        m1 = m2

        g0w0calc.context.timer.stop('W')

        for nQ, (bzq_c, symop) in enumerate(QSymmetryOp.get_symops(
                g0w0calc.wcalc.qd, iq, q_c)):

            for (progress, kpt1, kpt2)\
                    in g0w0calc.pair_distribution.kpt_pairs_by_q(bzq_c, 0, m2):
                pb.update((nQ + progress) / g0w0calc.wcalc.qd.mynk)

                k1 = g0w0calc.wcalc.gs.kd.bz2ibz_k[kpt1.K]
                i = g0w0calc.kpts.index(k1)
                g0w0calc.calculate_q(ie, i, kpt1, kpt2, qpdi, Wdict,
                                 symop=symop,
                                 sigmas=sigmas,
                                 blocks1d=blocks1d,
                                 pawcorr=pawcorr)

    for sigma in sigmas.values():
        sigma.sum(g0w0calc.context.comm)

    return sigmas

