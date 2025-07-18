import sys
sys.path.append("../")

from math import log, pi, sqrt
import numpy as np

from my_gpaw25.atom.configurations import configurations
from my_gpaw25.atom.radialgd import AERadialGridDescriptor
from my_gpaw25.xc import XC
from utils_debug_aecalc import *

class MyLDARadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, rgd, n_sLg, Y_L):
        nspins = len(n_sLg)
        n_sg = np.dot(Y_L, n_sLg)
        e_g = rgd.empty()
        dedn_sg = rgd.zeros(nspins)
        self.kernel.calculate(e_g, n_sg, dedn_sg) # call what?
        return e_g, dedn_sg


def my_calculate_spherical(xc, rgd, n_sg, v_sg, e_g=None):
    if e_g is None:
        e_g = rgd.empty()
    rcalc = MyLDARadialCalculator(xc.kernel)
    #print("Passed to rcalc = ", n_sg[:,np.newaxis])
    e_g[:], dedn_sg = rcalc(rgd, n_sg[:, np.newaxis], [1.0]) # will call __call__()
    v_sg[:] = dedn_sg
    return rgd.integrate(e_g) # return energy?


# ------------
# Main program
# ------------
symbol = "Si"

# Some default kwargs
xcname = "LDA_X+LDA_C_PW" # "LDA"
scalarrel = True
nofiles = True

# other kwargs
# XXX What are they?
gpernode = 150
corehole = None
orbital_free = False
tw_coeff = 1.0

# Get reference state:
Z, nlfe_j = configurations[symbol]
# Z is atomic number (integer)
# nlfe_j is a list containing tuples
# each tuples contain quantum number n, quantum number l,
# occupation, and energy (?)
# XXX: why is energy needed here? Probably for guess solution

print("All electron calculation for ", symbol)

# Collect principal quantum numbers, angular momentum quantum
# numbers, occupation numbers and eigenvalues (j is a combined
# index for n and l):
n_j = [n for n, l, f, e in nlfe_j]
l_j = [l for n, l, f, e in nlfe_j]
f_j = [f for n, l, f, e in nlfe_j]
e_j = [e for n, l, f, e in nlfe_j]

print("Electron configuration:")
print("Z = ", Z)
print("n_j = ", n_j)
print("l_j = ", l_j)
print("f_j = ", f_j)
print("e_j = ", e_j) # starting guess


maxnodes = max([n - l - 1 for n, l in zip(n_j, l_j)])
print("maxnodes = ", maxnodes)

N = (maxnodes + 1) * gpernode
beta = 0.4 # parameter for radial grid

# Core-hole stuffs setup here ....
# SKIPPED

# from run method

print("Radial grid points: ", N)
rgd = AERadialGridDescriptor(beta / N, 1.0 / N, N)

g = np.arange(N, dtype=float)
r = rgd.r_g
dr = rgd.dr_g
d2gdr2 = rgd.d2gdr2()

# Number of orbitals:
nj = len(n_j)
print("Number of orbitals: ", nj)

# Radial wave functions multiplied by radius:
u_j = np.zeros((nj,N))

# Effective potential multiplied by radius:
vr = np.zeros(N)

# Electron density:
n = np.zeros(N)

# Always spinpaired nspins=1
xc = XC(xcname)

vHr = np.zeros(N)
vXC = np.zeros(N)

# Initialize starting wavefunctions and calculate density from them
initialize_wave_functions(symbol, r, dr, l_j, e_j, u_j)
n[:] = calculate_density(r, f_j, u_j)
print("sum elec dens = ", np.sum(n))

niter = 0
NiterMax = 117
qOK = log(1e-10)
mix = 0.4

# orbital_free needs more iterations and coefficient
if orbital_free:
    mix = 0.01
    NiterMax = 2000
    e_j[0] /= tw_coeff
    if Z > 10:  # help convergence for third row elements
        mix = 0.002
        NiterMax = 10000


vrold = None

# calculate hartree potential
#radial_hartree(0, n * r * dr, r, vHr)
py_radial_hartree(0, n * r * dr, r, vHr)
print("sum vHr = ", np.sum(vHr))

# add potential from nuclear point charge (v = -Z / r)
vHr -= Z # vHr is vH times r, so the potential to added is (-Z/r)*r => -Z
print("sum vHr after adding -Z = ", np.sum(vHr))

# calculated exchange correlation potential and energy
vXC[:] = 0.0
print("sum n before my_calculate_spherical = ", np.sum(n))
Exc = my_calculate_spherical(xc, rgd, n.reshape((1, -1)), vXC.reshape((1, -1)))
print("sum vXC = ", np.sum(vXC))


