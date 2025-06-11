from my_gpaw25.xc import xc_string_to_dict
from ase.utils import basestring
from my_gpaw25.directmin.functional.fdpw.ks import KSFDPW
from my_gpaw25.directmin.functional.fdpw.pz import PZSICFDPW


def get_functional(func, *args):

    if isinstance(func, KSFDPW) or isinstance(func, PZSICFDPW):
        return func
    elif isinstance(func, basestring):
        func = xc_string_to_dict(func)

    if isinstance(func, dict):
        kwargs = func.copy()
        name = kwargs.pop('name').replace('-', '').lower()
        functional = {'ks': KSFDPW,
                      'pzsic': PZSICFDPW}[name](*args, **kwargs)
        return functional
    else:
        raise TypeError('Check functional parameter.')
