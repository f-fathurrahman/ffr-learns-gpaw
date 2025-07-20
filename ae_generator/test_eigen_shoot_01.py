import sys
sys.path.append("../")

from math import log, pi, sqrt
import numpy as np

from my_gpaw25.atom.configurations import configurations
from my_gpaw25.atom.radialgd import AERadialGridDescriptor
from my_gpaw25.xc import XC
from utils_debug_aecalc import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

def my_eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel, gmax=None):
    # us is modified

    print("\nENTER my_eigen_shoot")

    print(f"Integrating radial Schroedinger equation with energy = {e} and l={l}")

    if scalarrel:
        x = 0.5 * alpha**2  # x = 1 / (2c^2)
        Mr = r * (1.0 + x * e) - x * vr
    else:
        Mr = r
    c0 = l * (l + 1) + 2 * Mr * (vr - e * r)
    if gmax is None and np.all(c0 > 0):
        raise ConvergenceError("Bad initial electron density guess!")

    c1 = c10
    if scalarrel:
        c0 += x * r2dvdr / Mr
        c1 = c10 - x * r * r2dvdr / (Mr * dr)
    
    print("sum c10 = %18.10e" % np.sum(c10))
    print("sum c0 = %18.10e" % np.sum(c0))
    print("sum c1 = %18.10e" % np.sum(c1))
    print("sum c2 = %18.10e" % np.sum(c2))

    # vectors needed for numeric integration of diff. equation
    fm = 0.5 * c1 - c2
    fp = 0.5 * c1 + c2
    f0 = c0 - 2*c2

    print("sum fm = %18.10e" % np.sum(fm))
    print("sum fp = %18.10e" % np.sum(fp))
    print("sum f0 = %18.10e" % np.sum(f0))

    LARGE_U = 1e100
    SMALL_U = 1e-100

    if gmax is None:
        # set boundary conditions at r -> oo (u(oo) = 0 is implicit)
        u[-1] = 1.0
        #
        # perform backwards integration from infinity to the turning point
        g = len(u) - 2
        u[-2] = u[-1] * f0[-1] / fm[-1]
        #
        while c0[g] > 0.0:  # this defines the classical turning point
            # The update equation:
            u[g - 1] = (f0[g]*u[g] + fp[g]*u[g+1]) / fm[g]
            #
            if u[g - 1] < 0.0:
                # There should"t be a node here!  Use a more negative
                # eigenvalue:
                print("!!!!!!", end=" ")
                print("Early return in my_eigen_shoot 72")
                return 100, None
            # Don't let u become too large
            if u[g - 1] > LARGE_U:
                print("-------> !!!! u is scaled !!!!!")
                u *= SMALL_U
            g -= 1

        # stored values of the wavefunction and the first derivative
        # at the turning point
        gtp = g + 1
        utp = u[gtp]
        if gtp == len(u) - 1:
            return 100, 0.0
        dudrplus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    else:
        gtp = gmax

    print(f"my_eigen_shoot: after backward integration gtp={gtp}, c0={c0[gtp]} u={u[gtp]}")

    # set boundary conditions at r -> 0
    u[0] = 0.0 # interval size is 1 (for regular grid g)
    u[1] = 1.0
    # perform forward integration from zero to the turning point
    g = 1
    nodes = 0
    # integrate one step further than gtp
    # (such that dudr is defined in gtp)
    while g <= gtp:
        u[g + 1] = (fm[g]*u[g-1] - f0[g]*u[g]) / fp[g]
        if u[g+1] * u[g] < 0:
            print(f"Found nodes between: {g+1} and {g}")
            print(f"Values: {u[g+1]} and {u[g]}")
            nodes += 1
        g += 1
    # Early return
    if gmax is not None:
        print("Early return in my_eigen_shoot")
        return

    print(f"my_eigen_shoot: after inward integration u={u[gtp]}")

    # scale first part of wavefunction, such that it is continuous at gtp
    u[:gtp + 2] *= utp / u[gtp]
    
    # ffr: why not scale the second part? (result of backward integ)
    #u[gtp+2:] *= u[gtp] / utp
    # This can make new node: wavefunction will go negative

    print(f"my_eigen_shoot: after scale u={u[gtp]}")

    # determine size of the derivative discontinuity at gtp
    dudrminus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]

    print("utp = %18.10e" % utp)
    print("dudrplus = %18.10e" % dudrplus)
    print("dudrminus = %18.10e" % dudrminus)

    A = (dudrplus - dudrminus) * utp # utp factor is important
    # is A important for convergence?
    print("A = %18.10e" % A)
    print("nodes = ", nodes)

    print("EXIT my_eigen_shoot\n")

    return nodes, A



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

# solve Kohn-Sham equation and determine the density change
#solve_radial_schrod(rgd, N, r, dr, vr, d2gdr2, n_j, l_j,e_j, u_j, scalarrel=scalarrel)
#print("After solve_radial_schrod")
#print("e_j = ", e_j) # starting guess

# Begin debug eigen_shoot
c2 = -(r / dr)**2
c10 = -d2gdr2 * r**2  # first part of c1 vector

if scalarrel:
    r2dvdr = np.zeros(N)
    rgd.derivative(vr, r2dvdr)
    r2dvdr *= r
    r2dvdr -= vr
else:
    r2dvdr = None
print("sum abs r2dvdr = ", np.sum(np.abs(r2dvdr)))
# XXX: r2dvdr is local to this function
print("scalarrel = ", scalarrel)

ist = 3
RMAX_PLOT = 2.0 # should depend on ist for good visualization
# alternatively RMAX_PLOT can be determined from r[gtp] returned by eigen_shoot

# Set n, l, u
n = n_j[ist]
l = l_j[ist]
e = e_j[ist]
u = u_j[ist]
# Solutions for E:
# e_j =  [
#    -71.41258334942484,
#     -7.581580875185198, -6.289766502435535, -0.7733870397743262, -0.4539465426199744]

nn, A = my_eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)

#plt.clf()
#plt.plot(r, u, marker="o")
#plt.xlim(0.0, 0.2)
#plt.savefig("IMG_initial_integ_01.png", dpi=150)
#plt.clf()
#plt.plot(r, u, marker="o")
#plt.xlim(0.0, 10.0)
#plt.savefig("IMG_initial_integ_02.png", dpi=150)
#exit() # for debug


# adjust eigenenergy until u has the correct number of nodes
nodes = n - l - 1  # analytically expected number of nodes
delta = -0.2*e
while nn != nodes:
    diff = np.sign(nn - nodes)
    while diff == np.sign(nn - nodes):
        e -= diff * delta
        print("Same sign but nn != nodes")
        print(f"Integrate again with E = {e}")
        nn, A = my_eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)
    print("Different sign or the same no of nodes: nn={nn} nodes={nodes}")
    delta /= 2

print(f"Number of nodes is already good: nn={nn} nodes={nodes} diff={diff}")
print(f"   at E={e}")
# Note that A is not used here, only nn is used
exit()

norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
u *= 1.0 / sqrt(norm)

plt.plot(r, u); plt.xlim(0.0, RMAX_PLOT)
plt.savefig("IMG_after_node.png", dpi=150)

#print("nn = ", nn, " expected = ", nodes)
#print("A = ", A/sqrt(norm))


print("---Adjusting eigenenergy until u is smooth ----")

# adjust eigenenergy until u is smooth at the turning point
de = 1.0
iterNo = 1
while abs(de) > 1e-9:
    print(f"\niterNo={iterNo} e={e} de={de}")
    norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
    u *= 1.0 / sqrt(norm)
    print("norm_u = %18.10e" % norm)
    #
    plt.clf()
    plt.plot(r, u); plt.xlim(0.0, RMAX_PLOT)
    plt.savefig(f"IMG_iterNo_{iterNo}.png", dpi=150)
    #
    de = 0.5 * A / norm # here A is scaled by norm
    x = abs(de/e)
    print("x = ", x)
    if x > 0.1:
        de *= 0.1 / x
        print(f"de is adjusted to {de}")
    e -= de
    assert e < 0.0
    nn, A = my_eigen_shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel)
    #
    iterNo += 1
# Note that nn is not checked nor used in the above loop
# Only A is used


print(f"Final de={de}, e={e}")
print(f"nn = {nn} expected = {nodes}")
print(f"A = {A/sqrt(norm)}")

norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
u *= 1.0 / sqrt(norm)
plt.clf()
plt.plot(r, u); plt.xlim(0.0, RMAX_PLOT)
plt.savefig(f"IMG_iterNo_FINAL.png", dpi=150)

