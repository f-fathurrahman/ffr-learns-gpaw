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


def plot_radial_wavefuncs(r, fr, filename="IMG_psi.pdf"):
    plt.clf()
    plt.plot(r, fr)
    plt.grid(True)
    icut = search_icut(fr)
    rcut = r[icut]
    plt.xlim(0, rcut)
    plt.savefig(filename)


print() # MIT_MAGIC_COOKIE

beta = 0.4
for N in [10, 100, 1000, 2000, 5000]:
    ae_grid = MyAERadialGridDescriptor(beta/N, 1.0/N, N)
    r_g = ae_grid.r_g
    #f = R_10(r_g)
    #f = R_20(r_g)
    #f = R_30(r_g)
    #f = R_21(r_g)
    #f = R_31(r_g)
    #f = R_31_sympy(r_g)
    f = R_32(r_g)
    #f = R_32_sympy(r_g)
    res = ae_grid.integrate(f*f)/(4*pi) # need factor 4*pi
    print("N = {:8d}, res = {:18.10f}".format(N, res))


# Use the latest data
plot_radial_wavefuncs(r_g, f)
