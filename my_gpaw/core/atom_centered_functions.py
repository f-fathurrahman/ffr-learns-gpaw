from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import numpy as np
from my_gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from my_gpaw.mpi import serial_comm, MPIComm
from my_gpaw.kpt_descriptor import KPointDescriptor
from my_gpaw.lfc import LocalizedFunctionsCollection as LFC
from my_gpaw.spline import Spline
from my_gpaw.typing import ArrayLike2D, Array1D
if TYPE_CHECKING:
    from my_gpaw.core.uniform_grid import UniformGridFunctions


def to_spline(l: int,
              rcut: float,
              f: Callable[[Array1D], Array1D]) -> Spline:
    """Convert to GPAW's Spline object."""
    r = np.linspace(0, rcut, 100)
    return Spline(l, rcut, f(r))


class AtomCenteredFunctions:
    def __init__(self,
                 functions,
                 fracpos_ac: ArrayLike2D,
                 atomdist: AtomDistribution = None,
                 xp=None):
        self.xp = xp or np
        self.functions = [[to_spline(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self.fracpos_ac = np.array(fracpos_ac)
        self._atomdist = atomdist

        self._layout = None
        self._lfc = None

    def __repr__(self):
        funcs = [['spdfgh'[f.l] for f in ff] for ff in self.functions[:4]]
        if len(self.functions) > 4:
            funcs.append(...)
        return (f'{self.__class__.__name__}'
                f'(functions={funcs}, atomdist={self.atomdist})')

    @property
    def layout(self):
        self._lazy_init()
        return self._layout

    @property
    def atomdist(self):
        self._lazy_init()
        return self._atomdist

    def _lazy_init(self):
        raise NotImplementedError

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> AtomArrays:
        """Create AtomsArray for coefficients."""
        return self.layout.empty(dims, comm)

    def move(self, fracpos_ac, atomdist):
        """Move atoms to new positions."""
        self.fracpos_ac = np.array(fracpos_ac)
        self._lfc.set_positions(fracpos_ac, atomdist)

    def add_to(self, functions, coefs=1.0):
        """Add atom-centered functions multiplied by *coefs* to *functions*."""
        self._lazy_init()
        if isinstance(coefs, float):
            self._lfc.add(functions.data, coefs)
        else:
            self._lfc.add(functions.data, coefs, q=0)

    def integrate(self, functions, out=None):
        """Calculate integrals of atom-centered functions multiplied by
        *functions*.
        """
        self._lazy_init()
        if out is None:
            out = self.layout.empty(functions.dims, functions.comm)
        self._lfc.integrate(functions.data, out, q=0)
        return out

    def derivative(self, functions, out=None):
        """Calculate derivatives of integrals with respect to atom
        positions.
        """
        self._lazy_init()
        if out is None:
            out = self.layout.empty((3,) + functions.dims, functions.comm)
        coef_axiv = {a: self.xp.moveaxis(array_vxi, 0, -1)
                     for a, array_vxi in out._arrays.items()}
        self._lfc.derivative(functions.data, coef_axiv, q=0)
        return out

    def stress_tensor_contribution(self, a, c=1.0):
        return self._lfc.stress_tensor_contribution(a.data, c)


class UniformGridAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self,
                 functions,
                 fracpos_ac,
                 grid,
                 *,
                 atomdist=None,
                 integral=None,
                 cut=False,
                 xp=np):
        AtomCenteredFunctions.__init__(self,
                                       functions,
                                       fracpos_ac,
                                       atomdist, xp=xp)
        self.grid = grid
        self.integral = integral
        self.cut = cut

    def _lazy_init(self):
        if self._lfc is not None:
            return
        gd = self.grid._gd
        kd = KPointDescriptor(np.array([self.grid.kpt]))
        self._lfc = LFC(gd, self.functions, kd,
                        dtype=self.grid.dtype,
                        integral=self.integral,
                        forces=True,
                        cut=self.cut)
        self._lfc.set_positions(self.fracpos_ac)

        if self._atomdist is None:
            self._atomdist = AtomDistribution(
                ranks=np.array([sphere.rank for sphere in self._lfc.sphere_a]),
                comm=self.grid.comm)
        else:
            for sphere, rank in zip(self._lfc.sphere_a, self._atomdist.rank_a):
                assert sphere.rank == rank
            assert self.grid.comm is self._atomdist.comm

        self._layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                         for funcs in self.functions],
                                        self._atomdist,
                                        self.grid.dtype)

    def to_uniform_grid(self,
                        out: UniformGridFunctions,
                        scale: float = 1.0) -> UniformGridFunctions:
        out.data[:] = 0.0
        self.add_to(out, scale)
        return out
