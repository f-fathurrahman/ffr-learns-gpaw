import numpy as np


def fftshift(a_Q, axes):
    from my_gpaw25.gpu.cpupy import ndarray
    return ndarray(np.fft.fftshift(a_Q._data, axes=axes))


def ifftshift(a_Q, axes):
    from my_gpaw25.gpu.cpupy import ndarray
    return ndarray(np.fft.ifftshift(a_Q._data, axes=axes))
