import sys
sys.path.append("../")

from my_radial_grid import *

maxnodes = 2
gpernode = 150 # default?
beta = 0.4 # parameter for radial grid
N = gpernode*(maxnodes + 1)

ae_grid = MyAERadialGridDescriptor(beta/N, 1.0/N, N)

# Grid points in this case is
#                     a g
#            r(g) = -------,  g = 0, 1, ..., N - 1
#                   1 - b g

print("Grid parameters:")
print("a = ", ae_grid.a)
print("b = ", ae_grid.b)

# Grid points and derivative
NradialPoints = len(ae_grid)
r_g = ae_grid.r_g
dr_g = ae_grid.dr_g
d2gdr2 = ae_grid.d2gdr2()
for i in range(NradialPoints):
    print("%4d %18.10f %18.10f %18.10f" % (i+1, r_g[i], dr_g[i], d2gdr2[i]))


#print(ae_grid.dv_g)
#print(ae_grid.default_spline_points)

