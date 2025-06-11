"""Module defining  ``Eigensolver`` classes."""

from my_gpaw25.eigensolvers.rmmdiis import RMMDIIS
from my_gpaw25.eigensolvers.cg import CG
from my_gpaw25.eigensolvers.davidson import Davidson
from my_gpaw25.eigensolvers.direct import DirectPW
from my_gpaw25.lcao.eigensolver import DirectLCAO
from my_gpaw25.directmin.etdm_fdpw import FDPWETDM
from my_gpaw25.directmin.etdm_lcao import LCAOETDM


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
        if name == 'etdm':
            # Compatibility with old versions
            name = 'etdm-lcao'
        eigensolver = {'rmm-diis': RMMDIIS,
                       'cg': CG,
                       'dav': Davidson,
                       'lcao': DirectLCAO,
                       'direct': DirectPW,
                       'etdm-lcao': LCAOETDM,
                       'etdm-fdpw': FDPWETDM}[name](**eigensolver)

    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence.get('eigenstates', 4.0e-8)

    assert isinstance(eigensolver, DirectLCAO) == (mode.name == 'lcao') or \
           isinstance(eigensolver, LCAOETDM) == (mode.name == 'lcao')

    return eigensolver
