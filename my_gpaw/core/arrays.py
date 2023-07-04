from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Generic, TypeVar

import my_gpaw.fftw as fftw
import numpy as np
from ase.io.ulm import NDArrayReader
from my_gpaw.core.domain import Domain
from my_gpaw.core.matrix import Matrix
from my_gpaw.mpi import MPIComm
from my_gpaw.typing import Array1D

if TYPE_CHECKING:
    from my_gpaw.core.uniform_grid import UniformGridFunctions, UniformGrid

from my_gpaw.new import prod

DomainType = TypeVar('DomainType', bound=Domain)


class DistributedArrays(Generic[DomainType]):
    desc: DomainType

    def __init__(self,
                 dims: int | tuple[int, ...],
                 myshape: tuple[int, ...],
                 comm: MPIComm,
                 domain_comm: MPIComm,
                 data: np.ndarray | None,
                 dv: float,
                 dtype,
                 xp=None):
        self.myshape = myshape
        self.comm = comm
        self.domain_comm = domain_comm
        self.dv = dv

        # convert int to tuple:
        self.dims = dims if isinstance(dims, tuple) else (dims,)

        if self.dims:
            mydims0 = (self.dims[0] + comm.size - 1) // comm.size
            d1 = min(comm.rank * mydims0, self.dims[0])
            d2 = min((comm.rank + 1) * mydims0, self.dims[0])
            mydims0 = d2 - d1
            self.mydims = (mydims0,) + self.dims[1:]
        else:
            self.mydims = ()

        fullshape = self.mydims + self.myshape

        if data is not None:
            if data.shape != fullshape:
                raise ValueError(
                    f'Bad shape for data: {data.shape} != {fullshape}')
            if data.dtype != dtype:
                raise ValueError(
                    f'Bad dtype for data: {data.dtype} != {dtype}')
            if xp is not None:
                assert (xp is np) == isinstance(
                    data, (np.ndarray, NDArrayReader)), xp
        else:
            data = (xp or np).empty(fullshape, dtype)

        self.data = data
        self.xp: ModuleType
        if isinstance(data, (np.ndarray, NDArrayReader)):
            self.xp = np
        else:
            from my_gpaw.gpu import cupy as cp
            self.xp = cp
        self._matrix: Matrix | None = None

    def new(self, data=None) -> DistributedArrays:
        raise NotImplementedError

    def copy(self):
        return self.new(data=self.data.copy())

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        for index in range(self.dims[0]):
            yield self[index]

    def flat(self):
        if self.dims == ():
            yield self
        else:
            for index in np.indices(self.dims).reshape((len(self.dims), -1)).T:
                yield self[tuple(index)]

    def to_xp(self, xp):
        if xp is self.xp:
            # Disable assert for now as it would fail with our HIP-rfftn hack!
            # assert xp is np, 'cp -> cp should not be needed!'
            return self
        if xp is np:
            return self.new(data=self.xp.asnumpy(self.data))
        else:
            return self.new(data=xp.asarray(self.data))

    @property
    def matrix(self) -> Matrix:
        if self._matrix is not None:
            return self._matrix

        nx = prod(self.myshape)
        shape = (self.dims[0], prod(self.dims[1:]) * nx)
        myshape = (self.mydims[0], prod(self.mydims[1:]) * nx)
        dist = (self.comm, -1, 1)

        data = self.data.reshape(myshape)
        self._matrix = Matrix(*shape, data=data, dist=dist)

        return self._matrix

    def matrix_elements(self,
                        other,
                        *,
                        out: Matrix = None,
                        symmetric: bool = None,
                        function=None,
                        domain_sum=True,
                        cc: bool = False) -> Matrix:
        if symmetric is None:
            symmetric = self is other
        if function:
            other = function(other)

        M1 = self.matrix
        M2 = other.matrix
        out = M1.multiply(M2, opb='C', alpha=self.dv,
                          symmetric=symmetric, out=out)
        if not cc:
            out.complex_conjugate()
        # operate_and_multiply(self, self.layout.dv, out, function, ...)

        self._matrix_elements_correction(M1, M2, out, symmetric)

        if domain_sum:
            self.domain_comm.sum(out.data)

        return out

    def _matrix_elements_correction(self,
                                    M1: Matrix,
                                    M2: Matrix,
                                    out: Matrix,
                                    symmetric: bool) -> None:
        """Hook for PlaneWaveExpansion."""
        pass

    def abs_square(self,
                   weights: Array1D,
                   out: UniformGridFunctions) -> None:
        """Add weighted absolute square of data to output array.

        See also :xkcd:`849`.
        """
        raise NotImplementedError

    def gather(self, out=None, broadcast=False):
        raise NotImplementedError

    def interpolate(self,
                    plan1: fftw.FFTPlans = None,
                    plan2: fftw.FFTPlans = None,
                    grid: UniformGrid = None,
                    out: UniformGridFunctions = None) -> UniformGridFunctions:
        raise NotImplementedError


def operate_and_multiply(psit1, dv, out, operator, psit2):
    if psit1.comm:
        if psit2 is not None:
            assert psit2.comm is psit1.comm
        if psit1.comm.size > 1:
            out.comm = psit1.comm
            out.state = 'a sum is needed'

    comm = psit1.matrix.dist.comm
    N = len(psit1)
    n = (N + comm.size - 1) // comm.size
    mynbands = len(psit1.matrix.array)

    buf1 = psit1.new(nbands=n, dist=None)
    buf2 = psit1.new(nbands=n, dist=None)
    half = comm.size // 2
    psit = psit1.view(0, mynbands)
    if psit2 is not None:
        psit2 = psit2.view(0, mynbands)

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            n1 = min(rrank * n, N)
            n2 = min(n1 + n, N)
            if not (skip and comm.rank < half) and n2 > n1:
                rrequest = comm.receive(buf1.array[:n2 - n1], rrank, 11, False)
            if not (skip and comm.rank >= half) and len(psit1.array) > 0:
                srequest = comm.send(psit1.array, srank, 11, False)

        if r == 0:
            if operator:
                operator(psit1.array, psit2.array)
            else:
                psit2 = psit

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m12 = psit2.matrix_elements(psit, symmetric=(r == 0), cc=True,
                                        serial=True)
            n1 = min(((comm.rank - r) % comm.size) * n, N)
            n2 = min(n1 + n, N)
            out.array[:, n1:n2] = m12.array[:, :n2 - n1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                n1 = min(column * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    requests.append(
                        comm.send(out.array[:, n1:n2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                n1 = min(row * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    block = np.empty((mynbands, n2 - n1), out.dtype)
                    blocks.append((n1, n2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for n1, n2, block in blocks:
        out.array[:, n1:n2] = block
