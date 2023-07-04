import numpy as np


def get_goldstone_scaling(mode, chiks, Kxc_GG, dyson_solver):
    """Get kernel scaling parameter to fulfill a Goldstone condition."""
    assert chiks.qpd.kd.gamma,\
        r'The Goldstone condition is strictly bound to the Γ-point'

    # Find the frequency to determine the scaling from
    omega_w = chiks.zd.omega_w
    wgs = find_goldstone_frequency(mode, omega_w)

    # Only one rank, rgs, has the given frequency
    assert chiks.distribution == 'zGG',\
        'Only block distribution over frequencies is allowed at this point'
    wblocks = chiks.blocks1d
    rgs, mywgs = wblocks.find_global_index(wgs)

    # Let rgs find the rescaling
    fxcsbuf = np.empty(1, dtype=float)
    if wblocks.blockcomm.rank == rgs:
        chiks_GG = chiks.array[mywgs]
        fxcsbuf[:] = find_goldstone_scaling(mode, chiks_GG, Kxc_GG,
                                            dyson_solver)

    # Broadcast found rescaling
    wblocks.blockcomm.broadcast(fxcsbuf, rgs)
    fxcs = fxcsbuf[0]

    return fxcs


def find_goldstone_frequency(mode, omega_w):
    """Factory function for finding the appropriate frequency to determine
    the kernel scaling from according to different Goldstone criteria."""
    if mode == 'fm':
        return find_fm_goldstone_frequency(omega_w)
    elif mode == 'afm':
        return find_afm_goldstone_frequency(omega_w)
    else:
        raise ValueError(
            f"Allowed Goldstone scaling modes are 'fm', 'afm'. Got: {mode}")


def find_fm_goldstone_frequency(omega_w):
    """Find omega=0. as the fm Goldstone frequency."""
    wgs = np.abs(omega_w).argmin()
    if not np.allclose(omega_w[wgs], 0., atol=1.e-8):
        raise ValueError("Frequency grid needs to include"
                         + " omega=0. to allow 'fm' Goldstone scaling")

    return wgs


def find_afm_goldstone_frequency(omega_w):
    """Find the second smallest positive frequency
    as the afm Goldstone frequency."""
    # Set omega=0. and negative frequencies to np.inf
    omega1_w = np.where(omega_w < 1.e-8, np.inf, omega_w)
    # Sort for the two smallest positive frequencies
    omega2_w = np.partition(omega1_w, 1)
    # Find original index of second smallest positive frequency
    wgs = np.abs(omega_w - omega2_w[1]).argmin()

    return wgs


def find_goldstone_scaling(mode, chiks_GG, Kxc_GG, dyson_solver):
    """Factory function for finding the scaling of the kernel
    according to different Goldstone criteria."""
    assert mode in ['fm', 'afm'],\
        f"Allowed Goldstone scaling modes are 'fm', 'afm'. Got: {mode}"

    if mode == 'fm':
        return find_fm_goldstone_scaling(chiks_GG, Kxc_GG, dyson_solver)
    elif mode == 'afm':
        return find_afm_goldstone_scaling(chiks_GG, Kxc_GG, dyson_solver)


def find_fm_goldstone_scaling(chiks_GG, Kxc_GG, dyson_solver):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic inverse enhancement function has a root in (q=0, omega=0)."""
    fxcs = 1.
    kappaM = calculate_macroscopic_kappa(chiks_GG, Kxc_GG * fxcs, dyson_solver)
    # If kappaM > 0, increase scaling (recall: kappaM ~ 1 - Kxc Re{chi_0})
    scaling_incr = 0.1 * np.sign(kappaM)
    while abs(kappaM) > 1.e-7 and abs(scaling_incr) > 1.e-7:
        fxcs += scaling_incr
        if fxcs <= 0.0 or fxcs >= 10.:
            raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

        kappaM = calculate_macroscopic_kappa(
            chiks_GG, Kxc_GG * fxcs, dyson_solver)

        # If kappaM changes sign, change sign and refine increment
        if np.sign(kappaM) != np.sign(scaling_incr):
            scaling_incr *= -0.2

    return fxcs


def find_afm_goldstone_scaling(chiks_GG, Kxc_GG, dyson_solver):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic magnon spectrum vanishes at q=0. for finite frequencies."""
    fxcs = 1.
    SM = calculate_macroscopic_spectrum(chiks_GG, Kxc_GG * fxcs, dyson_solver)
    # If SM > 0., increase the scaling. If SM < 0., decrease the scaling.
    scaling_incr = 0.1 * np.sign(SM)
    while (SM < 0. or SM > 1.e-7) or abs(scaling_incr) > 1.e-7:
        fxcs += scaling_incr
        if fxcs <= 0. or fxcs >= 10.:
            raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

        SM = calculate_macroscopic_spectrum(
            chiks_GG, Kxc_GG * fxcs, dyson_solver)

        # If chi changes sign, change sign and refine increment
        if np.sign(SM) != np.sign(scaling_incr):
            scaling_incr *= -0.2

    return fxcs


def calculate_macroscopic_kappa(chiks_GG, Kxc_GG, dyson_solver):
    """Invert dyson equation and calculate the inverse enhancement function."""
    chi_GG = dyson_solver.invert_dyson_single_frequency(chiks_GG, Kxc_GG)

    return (chiks_GG[0, 0] / chi_GG[0, 0]).real


def calculate_macroscopic_spectrum(chiks_GG, Kxc_GG, dyson_solver):
    """Invert dyson equation and extract the macroscopic spectrum."""
    chi_GG = dyson_solver.invert_dyson_single_frequency(chiks_GG, Kxc_GG)

    return - chi_GG[0, 0].imag / np.pi
