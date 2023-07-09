from math import log, pi, sqrt
import numpy as np

from my_gpaw.atom.configurations import configurations
from my_gpaw.atom.radialgd import AERadialGridDescriptor
from my_gpaw.xc import XC

# Output: u_j is modified
def initialize_wave_functions(symbol, r, dr, l_j, e_j, u_j):
    r = r
    dr = dr
    # Initialize with Slater function:
    for l, e, u in zip(l_j, e_j, u_j):
        if symbol in ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']:
            a = sqrt(-4.0 * e)
        else:
            a = sqrt(-2.0 * e)
        u[:] = r**(1 + l) * np.exp(-a * r)
        norm = np.dot(u**2, dr)
        u *= 1.0 / sqrt(norm)
    return

def calculate_density(r, f_j, u_j):
    """Return the electron charge density divided by 4 pi"""
    n = np.dot(f_j, np.where(abs(u_j) < 1e-160, 0, u_j)**2) / (4 * pi)
    n[1:] /= r[1:]**2
    n[0] = n[1]
    return n



symbol = "Si"

# Some default kwargs
xcname = "LDA"
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
# XXX: why is energy needed here?


# Collect principal quantum numbers, angular momentum quantum
# numbers, occupation numbers and eigenvalues (j is a combined
# index for n and l):
n_j = [n for n, l, f, e in nlfe_j]
l_j = [l for n, l, f, e in nlfe_j]
f_j = [f for n, l, f, e in nlfe_j]
e_j = [e for n, l, f, e in nlfe_j]


maxnodes = max([n - l - 1 for n, l in zip(n_j, l_j)])
N = (maxnodes + 1) * gpernode
beta = 0.4


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

