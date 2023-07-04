"""Module defining  ``Eigensolver`` classes."""

from my_gpaw.eigensolvers.rmmdiis import RMMDIIS
from my_gpaw.eigensolvers.cg import CG
from my_gpaw.eigensolvers.davidson import Davidson
from my_gpaw.eigensolvers.direct import DirectPW
from my_gpaw.lcao.eigensolver import DirectLCAO
from my_gpaw.directmin.etdm import ETDM


def get_eigensolver(eigensolver, mode, convergence=None):
    """Create eigensolver object."""
    if eigensolver is None:
        if mode.name == 'lcao':
            eigensolver = 'lcao'
        else:
            eigensolver = 'dav'

    if isinstance(eigensolver, str):
        eigensolver = {'name': eigensolver}

    if isinstance(eigensolver, dict):
        eigensolver = eigensolver.copy()
        name = eigensolver.pop('name')
        eigensolver = {'rmm-diis': RMMDIIS,
                       'cg': CG,
                       'dav': Davidson,
                       'lcao': DirectLCAO,
                       'direct': DirectPW,
                       'etdm': ETDM,
                       }[name](**eigensolver)

    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence.get('eigenstates', 4.0e-8)

    assert isinstance(eigensolver, DirectLCAO) == (mode.name == 'lcao') or \
           isinstance(eigensolver, ETDM) == (mode.name == 'lcao')

    return eigensolver
