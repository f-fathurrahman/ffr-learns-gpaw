import sys
sys.path.append("../")

from math import log, pi, sqrt
import numpy as np

from my_gpaw25.atom.configurations import configurations
from my_gpaw25.atom.radialgd import AERadialGridDescriptor
from my_gpaw25.xc import XC
from utils_debug_aecalc import *

# ------------
# Main program
# ------------
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

while True:

    print(f"\nBegin iteration: {niter}")
    print("-----------------------")

    # calculate hartree potential
    #radial_hartree(0, n * r * dr, r, vHr)
    py_radial_hartree(0, n * r * dr, r, vHr)
    print("sum vHr = ", np.sum(vHr))

    # add potential from nuclear point charge (v = -Z / r)
    vHr -= Z # vHr is vH times r, so the potential to added is (-Z/r)*r => -Z
    print("sum vHr after adding -Z = ", np.sum(vHr))

    # calculated exchange correlation potential and energy
    vXC[:] = 0.0

    Exc = xc.calculate_spherical(rgd, n.reshape((1, -1)), vXC.reshape((1, -1)))

    # calculate new total Kohn-Sham effective potential and
    # admix with old version

    vr[:] = (vHr + vXC * r)
    print("sum abs vr = ", np.sum(np.abs(vr)))
    exit()

    if orbital_free:
        vr /= tw_coeff

    # Mix the potential
    if niter > 0:
        vr[:] = mix * vr + (1 - mix) * vrold
    vrold = vr.copy()

    # solve Kohn-Sham equation and determine the density change
    solve_radial_schrod(rgd, N, r, dr, vr, d2gdr2, n_j, l_j,e_j, u_j, scalarrel=scalarrel)
    dn = calculate_density(r, f_j, u_j) - n
    n += dn

    # estimate error from the square of the density change integrated
    q = log(np.sum((r * dn)**2))

    # print progress bar
    if niter == 0:
        q0 = q
        b0 = 0
    else:
        b = int((q0 - q) / (q0 - qOK) * 50)
        if b > b0:
            b0 = b

    # check if converged and break loop if so
    if q < qOK:
        print("Converged")
        break

    niter += 1
    if niter > NiterMax:
        raise RuntimeError("Did not converge!")


tau = radial_kinetic_energy_density(rgd, r, f_j, l_j, u_j)

print("Converged in %d iteration%s." % (niter, "s"[:niter != 1]))

Ekin = 0
for f, e in zip(f_j, e_j):
    Ekin += f * e

e_coulomb = 2 * pi * np.dot(n * r * (vHr - Z), dr)
Ekin += -4 * pi * np.dot(n * vr * r, dr)

if orbital_free:
    # e and vr are not scaled back
    # instead Ekin is scaled for total energy
    # (printed and inside setup)
    Ekin *= tw_coeff
    printt()
    printt("Lambda:{0}".format(tw_coeff))
    printt("Correct eigenvalue:{0}".format(e_j[0] * tw_coeff))
    printt()

print()
print("Energy contributions:")
print("-------------------------")
print("Kinetic:   %+13.6f" % Ekin)
print("XC:        %+13.6f" % Exc)
print("Potential: %+13.6f" % e_coulomb)
print("-------------------------")
print("Total:     %+13.6f" % (Ekin + Exc + e_coulomb))
ETotal = Ekin + Exc + e_coulomb
print()

print("state      eigenvalue         ekin         rmax")
print("-----------------------------------------------")
for m, l, f, e, u in zip(n_j, l_j, f_j, e_j, u_j):
    # Find kinetic energy: (numerical)
    k = e - np.sum((np.where(abs(u) < 1e-160, 0, u)**2 * vr * dr)[1:] / r[1:])

    # Find outermost maximum:
    g = N - 4
    while u[g - 1] >= u[g]:
        g -= 1
    x = r[g - 1:g + 2]
    y = u[g - 1:g + 2]
    A = np.transpose(np.array([x**i for i in range(3)]))
    c, b, a = np.linalg.solve(A, y)
    assert a < 0.0
    rmax = -0.5 * b / a

    s = "spdf"[l]
    print("%d%s^%-4.1f: %12.6f %12.6f %12.3f" % (m, s, f, e, k, rmax))
print("-----------------------------------------------")
print("(units: Bohr and Hartree)")


