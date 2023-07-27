from my_radial_grid import *
from hydrogenic_radial_wavefuncs import *
import matplotlib.pyplot as plt

# search from outermost point until QUITE_SMALL is exceeded
def search_icut(fr, QUITE_SMALL=1e-3):
    N = len(fr)
    icut = 0
    Nnodes = 0
    for i in range(N-1,0,-1):
        if abs(fr[i]) > QUITE_SMALL:
            icut = i
            break
    return icut



beta = 0.4
N = 5000
ae_grid = MyAERadialGridDescriptor(beta/N, 1.0/N, N)

r_g = ae_grid.r_g

f1 = R_32(r_g)
f2 = R_32_sympy(r_g)

plt.clf()
plt.plot(r_g, f1, label="orig")
plt.plot(r_g, f2, label="sympy")
plt.grid(True)

icut1 = search_icut(f1)
rcut1 = r_g[icut1]

icut2 = search_icut(f2)
rcut2 = r_g[icut2]

rcut = rcut1 if (rcut1 > rcut2) else rcut2
plt.xlim(0, rcut)
plt.savefig("IMG_debug_psi.pdf")


