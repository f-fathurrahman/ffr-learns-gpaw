import numpy as np

import my_gpaw25.cgpaw as cgpaw
from my_gpaw25.utilities.blas import axpy
from my_gpaw25 import gpu


def multi_axpy_cpu(a, x, y):
    for ai, xi, yi in zip(a, x, y):
        axpy(ai, xi, yi)


def multi_axpy(a, x, y):
    assert type(x) is type(y)

    if isinstance(a, (float, complex)):
        axpy(a, x, y)
    else:
        if not isinstance(x, np.ndarray):
            if not isinstance(a, np.ndarray):
                a_gpu = a
            else:
                a_gpu = gpu.copy_to_device(a)
            cgpaw.multi_axpy_gpu(gpu.get_pointer(a_gpu),
                                 a.dtype,
                                 gpu.get_pointer(x),
                                 x.shape,
                                 gpu.get_pointer(y),
                                 y.shape,
                                 x.dtype)
        else:
            multi_axpy_cpu(a, x, y)
