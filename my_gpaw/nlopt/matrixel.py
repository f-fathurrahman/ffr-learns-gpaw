from os import path

import numpy as np
from ase.utils.timing import Timer
from ase.parallel import parprint

from my_gpaw import GPAW
from my_gpaw.fd_operators import Gradient
from my_gpaw.mpi import world, broadcast, serial_comm
from my_gpaw.utilities.progressbar import ProgressBar


def get_mml(gs_name='gs.gpw', spin=0, ni=None, nf=None, timer=None):
    """Compute the momentum matrix elements.

    Input:
        gs_name         Ground state file name
        spin            Which spin channel (for spin-polarized systems 0 or 1)
        ni, nf          First and last band to compute the mml (0 to nb)
        timer           Timer to keep track of time
    Output:
        p_kvnn2         A big array in master
    """

    # Start the timer
    if timer is None:
        timer = Timer()
    parprint('Calculating momentum matrix elements...')

    # Load the ground state calculations
    with timer('Load the ground state'):
        parprint('Loading ground state data.')
        assert path.exists(
            gs_name), 'The gs file: {} does not exist!'.format(gs_name)
        calc = GPAW(
            gs_name, txt=None,
            parallel={'kpt': 1, 'band': 1},
            communicator=serial_comm)
        if calc.parameters['mode'] == 'lcao':
            calc.initialize_positions(calc.atoms)

    # Useful variables
    bzk_kc = calc.get_ibz_k_points()
    nbt = calc.get_number_of_bands()
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nbt
    nf = nbt + nf if (nf < 0) else nf
    blist = list(range(ni, nf))
    ns = calc.wfs.nspins
    assert spin < ns, 'Wrong spin input'
    nb = len(blist)
    nk = np.shape(bzk_kc)[0]
    cell_cv = calc.wfs.gd.cell_cv
    icell_cv = (2 * np.pi) * np.linalg.inv(cell_cv).T
    na = len(calc.atoms)
    rank = world.rank
    size = world.size
    est_mem = 2 * 3 * nk * nb**2 * 16 / 2**20
    parprint('At least {:.2f} MB of memory is required.'.format(est_mem))

    # Compute the matrix elements
    nkcore = int(np.ceil(nk / size))
    p_kvnn = np.zeros((nkcore, 3, nb, nb), dtype=complex)

    # if calc.parameters['mode'] == 'lcao':
    nabla_v = [
        Gradient(
            calc.wfs.gd, vv, 1.0, 4,
            complex).apply for vv in range(3)]
    phases = np.ones((3, 2), dtype=complex)

    # Initial call to print 0% progress
    ik = 0
    if world.rank == 0:
        pb = ProgressBar()

    # Loop over k points
    for ik in range(nkcore):
        k_ind = rank + size * ik
        if k_ind >= nk:
            break
        k_c = bzk_kc[k_ind]
        k_v = np.dot(k_c, icell_cv)

        # Get the wavefunctions
        with timer('Get wavefunctions and projections'):
            un_R = np.array(
                [calc.wfs.get_wave_function_array(
                    ib, k_ind, spin,
                    realspace=True,
                    periodic=True)
                    for ib in blist], complex)

            P_ani = []
            for ia in range(na):
                P_ani.append(calc.wfs.kpt_u[k_ind].P_ani[ia][blist])

        # Now compute the momentum part
        grad_nv = calc.wfs.gd.zeros((nb, 3), complex)
        with timer('Momentum calculation'):
            # Get the derivative
            for iv in range(3):
                for ib in range(nb):
                    nabla_v[iv](un_R[ib], grad_nv[ib, iv], phases)

            # Compute the integral
            p_vnn = np.transpose(
                calc.wfs.gd.integrate(un_R, grad_nv), (2, 0, 1))

            # Add the overlap part
            M_nn = np.array([calc.wfs.gd.integrate(
                un_R[ib], un_R) for ib in range(nb)])
            for iv in range(3):
                p_vnn[iv] += 1j * k_v[iv] * M_nn

        # The PAW corrections are added
        with timer('Add the PAW correction'):
            for ia in range(na):
                setup = calc.wfs.setups[ia]
                P0_ni = P_ani[ia]

                # Loop over components
                for iv in range(3):
                    tmp = setup.nabla_iiv[:, :, iv]
                    p_vnn[iv] += np.dot(
                        np.dot(P0_ni.conj(), tmp), P0_ni.T)

        # Make it momentum and store it
        p_kvnn[ik] = -1j * p_vnn

        # Print the progress
        if world.rank == 0:
            pb.update(ik / nkcore)
        ik += 1

    if world.rank == 0:
        pb.finish()

    # Gather all data to the master
    p_kvnn2 = []
    with timer('Gather the data to master'):
        parprint('Gathering date to the master.')
        recv_buf = None
        if rank == 0:
            recv_buf = np.empty((size, nkcore, 3, nb, nb),
                                dtype=complex)
        world.gather(p_kvnn, 0, recv_buf)
        if rank == 0:
            p_kvnn2 = np.zeros((nk, 3, nb, nb), dtype=complex)
            for ii in range(size):
                k_inds = range(ii, nk, size)
                p_kvnn2[k_inds] = recv_buf[ii, :len(k_inds)]

    # Print the timing
    if rank == 0:
        timer.write()

    return p_kvnn2


def make_nlodata(gs_name: str = 'gs.gpw',
                 out_name: str = 'mml.npz',
                 spin: str = 'all',
                 ni: int = 0,
                 nf: int = 0) -> None:
    """Get all required NLO data and store it in a file.

    Writes NLO data to file: w_sk, f_skn, E_skn, p_skvnn.

    Parameters:

    gs_name:
        Ground state file name
    out_name:
        Output filename
    spin:
        Which spin channel ('all', 's0' , 's1')
    ni:
        First band to compute the mml.
    nf:
        Last band to compute the mml (relative to number of bands
        for nf <= 0).

    """

    # Start the timer
    timer = Timer()

    # Get the energy and fermi levels (data is only in master)
    with timer('Get energies and fermi levels'):
        if world.rank == 0:
            # Load the ground state calculation
            calc = GPAW(gs_name, txt=None, communicator=serial_comm)

            # check the GS
            assert not calc.symmetry.point_group, \
                'Point group symmtery should be off.'

            nbt = calc.get_number_of_bands()
            if nf <= 0:
                nf += nbt
            ns = calc.wfs.nspins
            if spin == 'all':
                spins = list(range(ns))
            elif spin == 's0':
                spins = [0]
            elif spin == 's1':
                spins = [1]
                assert spins[0] < ns, 'Wrong spin input'
            else:
                raise NotImplementedError

            # Get the data
            w_sk = np.array([calc.get_k_point_weights() for s1 in spins])
            bz_vol = np.linalg.det(2 * np.pi * calc.wfs.gd.icell_cv)
            nk = len(w_sk[0])
            E_skn = np.array([calc.band_structure().todict()['energies'][s1]
                              for s1 in spins])
            f_skn = np.zeros((len(spins), nk, nbt), dtype=float)
            for sind, s1 in enumerate(spins):
                for ik in range(nk):
                    f_skn[sind, ik] = calc.get_occupation_numbers(
                        kpt=ik, spin=s1) / w_sk[sind, ik] * ns / 2.0
            w_sk *= bz_vol * 2.0 / ns
            broadcast(nf, root=0)
            broadcast(spins, root=0)
        else:
            nf = broadcast(None, root=0)
            spins = broadcast(None, root=0)

    # Compute the momentum matrix elements
    with timer('Compute the momentum matrix elements'):
        p_skvnn = []
        for s1 in spins:
            p_kvnn = get_mml(gs_name=gs_name, spin=s1,
                             ni=ni, nf=nf, timer=timer)
            p_skvnn.append(p_kvnn)

    # Save the output to the file
    if world.rank == 0:
        np.savez(out_name, w_sk=w_sk, f_skn=f_skn[:, :, ni:nf],
                 E_skn=E_skn[:, :, ni:nf], p_skvnn=np.array(p_skvnn, complex))


def get_rml(E_n, p_vnn, pol_v, Etol=1e-6):
    """
    Compute the position matrix elements

    Input:
        E_n             Energies
        p_vnn           Momentum matrix elements
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        r_vnn, D_vnn    Position and velocity difference matrix el.
    """

    # Useful variables
    nb = len(E_n)
    r_vnn = np.zeros((3, nb, nb), complex)
    D_vnn = np.zeros((3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    # Loop over components
    for v1 in set(pol_v):
        r_vnn[v1] = p_vnn[v1] / (1j * E_nn)
        r_vnn[v1, zeroind] = 0
        p_n = np.diag(p_vnn[v1])
        D_vnn[v1] = np.tile(p_n[:, None], (1, nb)) - \
            np.tile(p_n[None, :], (nb, 1))

    return r_vnn, D_vnn


def get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=1e-6):
    """
    Compute the generalized derivative of position matrix elements

    Input:
        E_n             Energies
        r_vnn           Momentum matrix elements
        D_vnn           Velocity difference
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        rd_vvnn         Generilized derivative of position
    """

    # Useful variables
    nb = len(E_n)
    rd_vvnn = np.zeros((3, 3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    for v1 in set(pol_v):
        for v2 in set(pol_v):
            tmp = (r_vnn[v1] * np.transpose(D_vnn[v2])
                   + r_vnn[v2] * np.transpose(D_vnn[v1])
                   + 1j * np.dot(r_vnn[v1], r_vnn[v2] * E_nn)
                   - 1j * np.dot(r_vnn[v2] * E_nn, r_vnn[v1])) / E_nn
            tmp[zeroind] = 0
            rd_vvnn[v1, v2] = tmp

    return rd_vvnn
