from my_gpaw.xc.libxc import LibXC
from my_gpaw.xc.lda import LDA
from my_gpaw.xc.gga import GGA
from my_gpaw.xc.mgga import MGGA
from my_gpaw.xc.noncollinear import NonCollinearLDAKernel
from my_gpaw import libraries


def xc_string_to_dict(string):
    """Convert XC specification string to dictionary.

    'name:key1=value1:...' -> {'name': <name>, key1: value1, ...}."""
    tokens = string.split(':')

    d = {'name': tokens[0]}
    for token in tokens[1:]:
        kw, val = token.split('=')
        # Convert value to int or float if possible
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        d[kw] = val
    return d


def XC(kernel, parameters=None, atoms=None, collinear=True):
    """Create XCFunctional object.

    kernel: XCKernel object, dict or str
        Kernel object or name of functional.
    parameters: ndarray
        Parameters for BEE functional.

    Recognized names are: LDA, PW91, PBE, revPBE, RPBE, BLYP, HCTH407,
    TPSS, M06-L, revTPSS, vdW-DF, vdW-DF2, EXX, PBE0, B3LYP, BEE,
    GLLBSC.  One can also use equivalent libxc names, for example
    GGA_X_PBE+GGA_C_PBE is equivalent to PBE, and LDA_X to the LDA exchange.
    In this way one has access to all the functionals defined in libxc.
    See xc_funcs.h for the complete list.  """

    if isinstance(kernel, str):
        kernel = xc_string_to_dict(kernel)

    kwargs = {}
    if isinstance(kernel, dict):
        kwargs = kernel.copy()
        name = kwargs.pop('name')
        backend = kwargs.pop('backend', None)

        if backend == 'libvdwxc' or name == 'vdW-DF-cx':
            # Must handle libvdwxc before old vdw implementation to override
            # behaviour for 'name'.  Also, cx is not implemented by the old
            # vdW module, so that always refers to libvdwxc.
            from my_gpaw.xc.libvdwxc import get_libvdwxc_functional
            return get_libvdwxc_functional(name=name, **kwargs)
        elif backend == 'ri':
            # Note: It is important that this if is before the next name is
            # HSExx, since otherwise PWHybrid would hijack the control flow.
            from my_gpaw.xc.ri import RI
            return RI(name, **kwargs)
        elif backend == 'pw' or name in ['HSE03', 'HSE06']:
            from my_gpaw.hybrids import HybridXC
            return HybridXC(name, **kwargs)
        elif backend:
            raise ValueError(
                'A special backend for the XC functional was given, '
                'but not understood. Please check if there is a typo.')

        if name in ['vdW-DF', 'vdW-DF2', 'optPBE-vdW', 'optB88-vdW',
                    'C09-vdW', 'mBEEF-vdW', 'BEEF-vdW']:
            from my_gpaw.xc.vdw import VDWFunctional
            return VDWFunctional(name, **kwargs)
        elif name in ['EXX', 'PBE0', 'B3LYP',
                      'CAMY-BLYP', 'CAMY-B3LYP', 'LCY-BLYP', 'LCY-PBE']:
            from my_gpaw.xc.hybrid import HybridXC
            return HybridXC(name, **kwargs)
        elif name.startswith('LCY-') or name.startswith('CAMY-'):
            parts = name.split('(')
            from my_gpaw.xc.hybrid import HybridXC
            return HybridXC(parts[0], omega=float(parts[1][:-1]))
        elif name == 'BEE1':
            from my_gpaw.xc.bee import BEE1
            kernel = BEE1(parameters)
        elif name == 'BEE2':
            from my_gpaw.xc.bee import BEE2
            kernel = BEE2(parameters)
        elif name.startswith('GLLB'):
            from my_gpaw.xc.gllb.nonlocalfunctionalfactory import (
                get_nonlocal_functional)
            xc = get_nonlocal_functional(name, **kwargs)
            return xc
        elif name == 'LB94':
            from my_gpaw.xc.lb94 import LB94
            kernel = LB94()
        elif name == 'TB09':
            from my_gpaw.xc.tb09 import TB09
            return TB09(**kwargs)
        elif name.endswith('PZ-SIC'):
            from my_gpaw.xc.sic import SIC
            return SIC(xc=name[:-7], **kwargs)
        elif name in {'TPSS', 'revTPSS', 'M06-L'}:
            assert libraries['libxc'], 'Please compile with libxc'
            from my_gpaw.xc.kernel import XCKernel
            kernel = XCKernel(name)
        elif name in {'LDA', 'PBE', 'revPBE', 'RPBE', 'PW91'}:
            from my_gpaw.xc.kernel import XCKernel
            kernel = XCKernel(name)
        elif name.startswith('old'):
            from my_gpaw.xc.kernel import XCKernel
            kernel = XCKernel(name[3:])
        elif name == 'PPLDA':
            from my_gpaw.xc.lda import PurePythonLDAKernel
            kernel = PurePythonLDAKernel()
        elif name in ['pyPBE', 'pyPBEsol', 'pyRPBE', 'pyzvPBEsol']:
            from my_gpaw.xc.gga import PurePythonGGAKernel
            kernel = PurePythonGGAKernel(name)
        elif name == '2D-MGGA':
            from my_gpaw.xc.mgga import PurePython2DMGGAKernel
            kernel = PurePython2DMGGAKernel(name, parameters)
        elif name[0].isdigit():
            from my_gpaw.xc.parametrizedxc import ParametrizedKernel
            kernel = ParametrizedKernel(name)
        elif name == 'null':
            from my_gpaw.xc.kernel import XCNull
            kernel = XCNull()
        elif name == 'QNA':
            from my_gpaw.xc.qna import QNA
            return QNA(atoms, kernel['parameters'], kernel['setup_name'],
                       alpha=kernel['alpha'], stencil=kwargs.get('stencil', 2))
        else:
            kernel = LibXC(name)

    if kernel.type == 'LDA':
        if not collinear:
            kernel = NonCollinearLDAKernel(kernel)
        xc = LDA(kernel, **kwargs)
        return xc

    elif kernel.type == 'GGA':
        return GGA(kernel, **kwargs)
    else:
        return MGGA(kernel, **kwargs)
