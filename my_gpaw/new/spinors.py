from __future__ import annotations

import numpy as np
from my_gpaw.core.domain import Domain
from my_gpaw.typing import Vector


class SpinorWaveFunctionDescriptor(Domain):
    def __init__(self,
                 desc: Domain,
                 qspiral_v: Vector = None):
        self.desc = desc
        self.qspiral_v = (np.asarray(qspiral_v) if qspiral_v is not None else
                          None)
        Domain.__init__(self, desc.cell_cv, desc.pbc_c, desc.kpt_c, desc.comm,
                        complex)
        self.myshape = (2,) + desc.myshape
        self.itemsize = desc.itemsize

    def __repr__(self):
        return f'{self.__class__.__name__}({self.desc}, {self.qspiral_v})'

    def new(self, *, kpt):
        desc = self.desc.new(kpt=kpt)
        desc.qspiral_v = self.qspiral_v
        return SpinorWaveFunctionDescriptor(desc, self.qspiral_v)

    def empty(self, nbands, band_comm, xp=None):
        return self.desc.empty((nbands, 2), band_comm)

    def global_shape(self) -> tuple[int, ...]:
        return (2,) + self.desc.global_shape()
