import numpy as np
from my_gpaw25.core import UGDesc

a = 4.0
n = 20 # sampling points, same for all three directions
grid = UGDesc(cell=a*np.eye(3), size=(n,n,n))

# given UGDesc instance, we can create UGArray instances
func_R = grid.empty()
print("func_R shape = ", func_R.data.shape)
# func_R.data is a Numpy array
# assign
func_R.data[:] = 1.0

# "multidimensional" UGArray
func32 = grid.zeros((3,2))
print("shape func32 = ", func32.data.shape)
