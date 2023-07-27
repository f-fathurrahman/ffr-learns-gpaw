import matplotlib.pyplot as plt
from hydrogenic_radial_wavefuncs import *
from my_radial_grid import *

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
N = 150*(2 + 1)
ae_grid = MyAERadialGridDescriptor(beta/N, 1.0/N, N)

r_g = ae_grid.r_g
dr_g = ae_grid.dr_g

# evaluate psi
#psi1 = R_10(r_g)
#psi1 = R_30(r_g)
psi1 = R_32(r_g)
icut = search_icut(psi1)
rcut = r_g[icut]

plt.clf()
plt.plot(r_g, psi1)
plt.xlim(0.0, rcut)
plt.grid(True)
plt.savefig("IMG_psi1.pdf")
