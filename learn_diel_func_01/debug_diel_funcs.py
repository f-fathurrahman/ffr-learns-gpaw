import numpy as np
from my_df import Chi0DysonEquations, CoulombKernel, DensityXCKernel

def debug_get_chi0_dyson_eqs(
        diel_func,
        q_c: list | np.ndarray = [0, 0, 0],
        truncation: str | None = None,
        xc: str = 'RPA',
        **xckwargs
    ) -> Chi0DysonEquations:
    """Set up the Dyson equation for χ(q,ω) at given wave vector q.

    Parameters
    ----------
    truncation : str or None
        Truncation of the Hartree kernel.
    xc : str
        Exchange-correlation kernel for LR-TDDFT calculations.
        If xc == 'RPA', the dielectric response is treated in the random
        phase approximation.
    **xckwargs
        Additional parameters for the chosen xc kernel.
    """
    print("q_c = ", q_c)
    chi0 = diel_func.get_chi0(q_c)
    #
    coulomb = CoulombKernel.from_gs(diel_func.gs, truncation=truncation)
    #
    print("xc = ", xc)
    if xc == 'RPA':
        xc_kernel = None
    else:
        xc_kernel = DensityXCKernel.from_functional(
            diel_func.gs, diel_func.context, functional=xc, **xckwargs)
    #
    return Chi0DysonEquations(chi0, coulomb, xc_kernel, diel_func.gs.cd)