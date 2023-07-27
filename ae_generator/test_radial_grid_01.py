from my_radial_grid import *

eq_grid = MyEquidistantRadialGridDescriptor(0.01, N=100, h0=0.0)

print(eq_grid.r_g)
print(eq_grid.dr_g)
print(eq_grid.N)
print(eq_grid.dv_g)
print(eq_grid.default_spline_points)