import numpy as np
from my_gpaw25.core import UGDesc
from my_gpaw25.core import PWDesc

a = 4.0
n = 20 # sampling points, same for all three directions
grid = UGDesc(cell=a*np.eye(3), size=(n,n,n))
func_R = grid.empty()
func_R.data[:] = 1.0

pw = PWDesc(ecut=100, cell=grid.cell)

func_G = pw.empty()
func_R.fft(out=func_G)

# reciprocal vectors
G = pw.reciprocal_vectors()

# TODO: test mapping indices