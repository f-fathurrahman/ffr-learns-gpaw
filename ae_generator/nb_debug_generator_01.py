import sys
sys.path.append("../")

from math import pi, sqrt
import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt

from my_gpaw25.atom.all_electron import AllElectron, shoot
from my_gpaw25.utilities import hartree
from my_gpaw25.atom.filter import Filter
from my_gpaw25 import __version__ as version
from my_gpaw25.setup_data import SetupData

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

from utils_debug_generator import *

# AllElectron object
symbol = "Pd"

xcname = "LDA"
scalarrel = True
corehole = None
configuration = None
nofiles = True
txt = "-"
gpernode = 150
orbital_free = False
tw_coeff = 1.0


ae_calc = AllElectron(symbol,
    xcname = "LDA",
    scalarrel = scalarrel,
    corehole = corehole,
    configuration = configuration,
    nofiles = nofiles,
    txt = txt,
    gpernode = gpernode,
    orbital_free = orbital_free,
    tw_coeff = tw_coeff,
)

from my_gpaw.atom.configurations import parameters, configurations
par = parameters[symbol]

par_keys = par.keys()
# Some kwargs for Generator.run() method
core = par["core"]
rcut = par["rcut"]
#XXX Those two parameters are always present for all atomic species?

if "extra" in par_keys:
    extra = par["extra"]
else:
    extra = None

if "vbar" in par_keys:
    vbar = par["vbar"]
else:
    vbar = None

if "filter" in par_keys:
    hfilter, xfilter = par["filter"]
else:
    hfilter = 0.4
    xfilter = 1.75
#
print(f"hfilter = {hfilter} xfilter = {xfilter}")

if "rcutcomp" in par_keys:
    rcutcomp = par["rcutcomp"]
else:
    rcutcomp = None
print("rcutcomp = ", rcutcomp)

if "empty_states" in par_keys:
    empty_states = par["empty_states"]
else:
    empty_states = ""

exx = False
name = None
normconserving = ""
write_xml = True
yukawa_gamma = 0.0
logderiv = False



if isinstance(rcut, float):
    rcut_l = [rcut]
else:
    rcut_l = rcut
rcutmax = max(rcut_l)
rcutmin = min(rcut_l)

print("rcut_l = ", rcut_l)
print("rcutmax = ", rcutmax)
print("rcutmin = ", rcutmin)


if rcutcomp is None:
    rcutcomp = rcutmin


# Get some parameters from Generator superclass (i.e. AllElectron)
Z = ae_calc.Z
n_j = ae_calc.n_j
l_j = ae_calc.l_j
f_j = ae_calc.f_j
e_j = ae_calc.e_j


if vbar is None:
    vbar = ("poly", rcutmin * 0.9)
vbar_type, rcutvbar = vbar

normconserving_l = [x in normconserving for x in "spdf"]
print(normconserving_l)

# Parse core string:
j = 0
if core.startswith("["):
    a, core = core.split("]")
    core_symbol = a[1:]
    j = len(configurations[core_symbol][1])

while core != "":
    assert n_j[j] == int(core[0])
    assert l_j[j] == "spdf".find(core[1])
    if j != ae_calc.jcorehole:
        assert f_j[j] == 2 * (2 * l_j[j] + 1)
    j += 1
    core = core[2:]

njcore = j
print("njcore = ", njcore)

while empty_states != "":
    n = int(empty_states[0])
    l = "spdf".find(empty_states[1])
    assert n == 1 + l + l_j.count(l)
    n_j.append(n)
    l_j.append(l)
    f_j.append(0.0)
    e_j.append(-0.01)
    empty_states = empty_states[2:]

if 2 in l_j[njcore:]:
    print("We have a bound valence d-state.")
    print("Add bound s- and p-states if not already there.")
    for l in [0, 1]:
        if l not in l_j[njcore:]:
            n_j.append(1 + l + l_j.count(l))
            l_j.append(l)
            f_j.append(0.0)
            e_j.append(-0.01)

if l_j[njcore:] == [0] and Z > 2:
    # We have only a bound valence s-state and we are not
    # hydrogen and not helium.  Add bound p-state:
    n_j.append(n_j[njcore])
    l_j.append(1)
    f_j.append(0.0)
    e_j.append(-0.01)

nj = len(n_j)

Nv = sum(f_j[njcore:])
Nc = sum(f_j[:njcore])

lmaxocc = max(l_j[njcore:])
lmax = max(l_j[njcore:])

#  Parameters for orbital_free
if orbital_free:
    n_j = [1]
    l_j = [0]
    f_j = [Z]
    e_j = [e_j[0]]
    nj = len(n_j)
    lmax = 0
    lmaxocc = 0

# Do all-electron calculation:
ae_calc.run()

# Highest occupied atomic orbital:
emax = max(e_j)

N = ae_calc.N
r = ae_calc.r
dr = ae_calc.dr
d2gdr2 = ae_calc.d2gdr2
beta = ae_calc.beta

dv = r**2 * dr

print()
print("Generating PAW setup")
if core != "":
    print("Frozen core:", core)

# So far - no ghost-states:
ghost = False


# Calculate the kinetic energy of the core states:
Ekincore = 0.0
j = 0
for f, e, u in zip(f_j[:njcore], e_j[:njcore], ae_calc.u_j[:njcore]):
    u = np.where(abs(u) < 1e-160, 0, u)  # XXX Numeric!
    k = e - np.sum((u**2 * ae_calc.vr * dr)[1:] / r[1:])
    Ekincore += f * k
    if j == ae_calc.jcorehole:
        Ekincorehole = k # self
    j += 1

# Calculate core density:
if njcore == 0:
    nc = np.zeros(N)
else:
    uc_j = ae_calc.u_j[:njcore]
    uc_j = np.where(abs(uc_j) < 1e-160, 0, uc_j)  # XXX Numeric!
    nc = np.dot(f_j[:njcore], uc_j**2) / (4 * pi)
    nc[1:] /= r[1:]**2
    nc[0] = nc[1]

# print("nc = ", nc)
# self.nc = nc

# Calculate core kinetic energy density
if njcore == 0:
    tauc = np.zeros(N)
else:
    tauc = ae_calc.radial_kinetic_energy_density(f_j[:njcore], l_j[:njcore], ae_calc.u_j[:njcore])
    print("Kinetic energy of the core from tauc =", np.dot(tauc * r * r, dr) * 4 * pi)


# Order valence states with respect to angular momentum
# quantum number:
n_ln = n_ln = []
f_ln = f_ln = []
e_ln = e_ln = []
for l in range(lmax + 1):
    n_n = []
    f_n = []
    e_n = []
    for j in range(njcore, nj):
        if l_j[j] == l:
            n_n.append(n_j[j])
            f_n.append(f_j[j])
            e_n.append(e_j[j])
    n_ln.append(n_n)
    f_ln.append(f_n)
    e_ln.append(e_n)

print("n_ln = ", n_ln)
print("f_ln = ", f_ln)
print("e_ln = ", e_ln)


# Add extra projectors:
if extra is not None:
    if len(extra) == 0:
        lmaxextra = 0
    else:
        lmaxextra = max(extra.keys())
    if lmaxextra > lmax:
        for l in range(lmax, lmaxextra):
            n_ln.append([])
            f_ln.append([])
            e_ln.append([])
        lmax = lmaxextra
    for l in extra:
        nn = -1
        for e in extra[l]:
            n_ln[l].append(nn)
            f_ln[l].append(0.0)
            e_ln[l].append(e)
            nn -= 1
else:
    print()
    print("Automatic number of projectors")
    print()
    # Automatic:
    # Make sure we have two projectors for each occupied channel:
    for l in range(lmaxocc + 1):
        if len(n_ln[l]) < 2 and not normconserving_l[l]:
            # Only one - add one more:
            assert len(n_ln[l]) == 1
            n_ln[l].append(-1)
            f_ln[l].append(0.0)
            e_ln[l].append(1.0 + e_ln[l][0])
    if lmaxocc < 2 and lmaxocc == lmax:
        # Add extra projector for l = lmax + 1:
        n_ln.append([-1])
        f_ln.append([0.0])
        e_ln.append([0.0])
        lmax += 1


print("lmax = ", lmax)
#self.lmax = lmax

rcut_l.extend([rcutmin] * (lmax + 1 - len(rcut_l)))
print("Cutoffs:")
for rc, s in zip(rcut_l, "spdf"):
    print("rc(%s)=%.3f" % (s, rc))
print("rc(vbar)=%.3f" % rcutvbar)
print("rc(comp)=%.3f" % rcutcomp)
print("rc(nct)=%.3f" % rcutmax)
print()
print("Kinetic energy of the core states: %.6f" % Ekincore)


# Allocate arrays:
u_ln = []  # phi * r
s_ln = []  # phi-tilde * r
q_ln = []  # p-tilde * r
for l in range(lmax + 1):
    nn = len(n_ln[l])
    u_ln.append(np.zeros((nn, N)))
    s_ln.append(np.zeros((nn, N)))
    q_ln.append(np.zeros((nn, N)))

# Fill in all-electron wave functions:
for l in range(lmax + 1):
    # Collect all-electron wave functions:
    u_n = [ae_calc.u_j[j] for j in range(njcore, nj) if l_j[j] == l]
    for n, u in enumerate(u_n):
        u_ln[l][n] = u

# Grid-index corresponding to rcut:
gcut_l = [1 + int(rc * N / (rc + beta)) for rc in rcut_l]

rcutfilter = xfilter * rcutmax
gcutfilter = 1 + int(rcutfilter * N / (rcutfilter + beta))
gcutmax = 1 + int(rcutmax * N / (rcutmax + beta))

# Outward integration of unbound states stops at 3 * rcut:
gmax = int(3 * rcutmax * N / (3 * rcutmax + beta))
assert gmax > gcutfilter

# Calculate unbound extra states:
c2 = -(r / dr)**2
c10 = -d2gdr2 * r**2
for l, (n_n, e_n, u_n) in enumerate(zip(n_ln, e_ln, u_ln)):
    for n, e, u in zip(n_n, e_n, u_n):
        if n < 0:
            u[:] = 0.0
            shoot(u, l, ae_calc.vr, e, ae_calc.r2dvdr, r, dr, c10, c2,
                  ae_calc.scalarrel, gmax=gmax)
            u *= 1.0 / u[gcut_l[l]]

charge = Z - Nv - Nc
print("Charge: %.1f" % charge)
print("Core electrons: %.1f" % Nc)
print("Valence electrons: %.1f" % Nv)


# Construct smooth wave functions:
coefs = []
for l, (u_n, s_n) in enumerate(zip(u_ln, s_ln)):
    nodeless = True
    gc = gcut_l[l]
    for u, s in zip(u_n, s_n):
        s[:] = u
        if normconserving_l[l]:
            A = np.zeros((5, 5))
            A[:4, 0] = 1.0
            A[:4, 1] = r[gc - 2:gc + 2]**2
            A[:4, 2] = A[:4, 1]**2
            A[:4, 3] = A[:4, 1] * A[:4, 2]
            A[:4, 4] = A[:4, 2]**2
            A[4, 4] = 1.0
            a = u[gc - 2:gc + 3] / r[gc - 2:gc + 3]**(l + 1)
            a = np.log(a)
            #
            def f(x):
                a[4] = x
                b = np.linalg.solve(A, a)
                r1 = r[:gc]
                r2 = r1**2
                rl1 = r1**(l + 1)
                y = b[0] + r2 * (b[1] + r2 * (b[2] + r2 *(b[3] + r2 * b[4])))
                y = np.exp(y)
                s[:gc] = rl1 * y
                return np.dot(s**2, dr) - 1
            x1 = 0.0
            x2 = 0.001
            f1 = f(x1)
            f2 = f(x2)
            while abs(f1) > 1e-6:
                x0 = (x1 / f1 - x2 / f2) / (1 / f1 - 1 / f2)
                f0 = f(x0)
                if abs(f1) < abs(f2):
                    x2, f2 = x1, f1
                x1, f1 = x0, f0
        #
        else:
            A = np.ones((4, 4))
            A[:, 0] = 1.0
            A[:, 1] = r[gc - 2:gc + 2]**2
            A[:, 2] = A[:, 1]**2
            A[:, 3] = A[:, 1] * A[:, 2]
            a = u[gc - 2:gc + 2] / r[gc - 2:gc + 2]**(l + 1)
            if 0:  # l < 2 and nodeless:
                a = np.log(a)
            a = np.linalg.solve(A, a)
            r1 = r[:gc]
            r2 = r1**2
            rl1 = r1**(l + 1)
            y = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * (a[3])))
            if 0:  # l < 2 and nodeless:
                y = np.exp(y)
            s[:gc] = rl1 * y
        #
        coefs.append(a)
        if nodeless:
            if not np.all(s[1:gc] > 0.0):
                raise RuntimeError("Error: The %d%s pseudo wave has a node!" % (n_ln[l][0], "spdf"[l]))
            # Only the first state for each l must be nodeless:
            nodeless = False

# Calculate pseudo core density:
gcutnc = 1 + int(rcutmax * N / (rcutmax + beta))
nct = nc.copy()
A = np.ones((4, 4))
A[0] = 1.0
A[1] = r[gcutnc - 2:gcutnc + 2]**2
A[2] = A[1]**2
A[3] = A[1] * A[2]
a = nc[gcutnc - 2:gcutnc + 2]
a = np.linalg.solve(np.transpose(A), a)
r2 = r[:gcutnc]**2
nct[:gcutnc] = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * a[3]))
print("Pseudo-core charge: %.6f" % (4 * pi * np.dot(nct, dv)))


# ... and the pseudo core kinetic energy density:
tauct = tauc.copy()
a = tauc[gcutnc - 2:gcutnc + 2]
a = np.linalg.solve(np.transpose(A), a)
tauct[:gcutnc] = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * a[3]))

# ... and the soft valence density:
nt = np.zeros(N)
for f_n, s_n in zip(f_ln, s_ln):
    nt += np.dot(f_n, s_n**2) / (4 * pi)
nt[1:] /= r[1:]**2
nt[0] = nt[1]
nt += nct
#self.nt = nt

# Calculate the shape function:
x = r / rcutcomp
gaussian = np.zeros(N)
gamma = 10.0
gaussian[:gmax] = np.exp(-gamma * x[:gmax]**2)
gt = 4 * (gamma / rcutcomp**2)**1.5 / sqrt(pi) * gaussian
print("Shape function alpha=%.3f" % (gamma / rcutcomp**2))
norm = np.dot(gt, dv)
#  print norm, norm-1
assert abs(norm - 1) < 1e-2
gt /= norm


# Calculate smooth charge density:
Nt = np.dot(nt, dv)
rhot = nt - (Nt + charge / (4 * pi)) * gt
print("Pseudo-electron charge", 4 * pi * Nt)

vHt = np.zeros(N)
hartree(0, rhot * r * dr, r, vHt)
vHt[1:] /= r[1:]
vHt[0] = vHt[1]

vXCt = np.zeros(N)
#extra_xc_data = {} # only for GLLB

ae_calc.xc.calculate_spherical(ae_calc.rgd, nt.reshape((1, -1)), vXCt.reshape((1, -1)))

vt = vHt + vXCt

if orbital_free:
    vt /= tw_coeff

# Construct zero potential:
gc = 1 + int(rcutvbar * N / (rcutvbar + beta))
if vbar_type == "f":
    assert lmax == 2
    uf = np.zeros(N)
    l = 3
    #
    # Solve for all-electron f-state:
    eps = 0.0
    shoot(uf, l, ae_calc.vr, eps, ae_calc.r2dvdr, r, dr, c10, c2, scalarrel, gmax=gmax)
    uf *= 1.0 / uf[gc]
    #
    # Fit smooth pseudo f-state polynomium:
    A = np.ones((4, 4))
    A[:, 0] = 1.0
    A[:, 1] = r[gc - 2:gc + 2]**2
    A[:, 2] = A[:, 1]**2
    A[:, 3] = A[:, 1] * A[:, 2]
    a = uf[gc - 2:gc + 2] / r[gc - 2:gc + 2]**(l + 1)
    a0, a1, a2, a3 = np.linalg.solve(A, a)
    r1 = r[:gc]
    r2 = r1**2
    rl1 = r1**(l + 1)
    y = a0 + r2 * (a1 + r2 * (a2 + r2 * a3))
    sf = uf.copy()
    sf[:gc] = rl1 * y
    #
    # From 0 to gc, use analytic formula for kinetic energy operator:
    r4 = r2**2
    r6 = r4 * r2
    enumerator = (a0 * l * (l + 1) +
                  a1 * (l + 2) * (l + 3) * r2 +
                  a2 * (l + 4) * (l + 5) * r4 +
                  a3 * (l + 6) * (l + 7) * r6)
    denominator = a0 + a1 * r2 + a2 * r4 + a3 * r6
    ekin_over_phit = - 0.5 * (enumerator / denominator - l * (l + 1))
    ekin_over_phit[1:] /= r2[1:]
    #
    vbar = eps - vt
    vbar[:gc] -= ekin_over_phit
    vbar[0] = vbar[1]  # Actually we can collect the terms into
    # a single fraction without poles, so as to avoid doing this,
    # but this is good enough
    #
    # From gc to gmax, use finite-difference formula for kinetic
    # energy operator:
    vbar[gc:gmax] -= ae_calc.kin(l, sf)[gc:gmax] / sf[gc:gmax] # ???
    vbar[gmax:] = 0.0
else:
    assert vbar_type == "poly"
    A = np.ones((2, 2))
    A[0] = 1.0
    A[1] = r[gc - 1:gc + 1]**2
    a = vt[gc - 1:gc + 1]
    a = np.linalg.solve(np.transpose(A), a)
    r2 = r**2
    vbar = a[0] + r2 * a[1] - vt
    vbar[gc:] = 0.0
#
vt += vbar


# Construct projector functions:
for l, (e_n, s_n, q_n) in enumerate(zip(e_ln, s_ln, q_ln)):
    for e, s, q in zip(e_n, s_n, q_n):
        q[:] = ae_calc.kin(l, s) + (vt - e) * s
        q[gcutmax:] = 0.0
#
my_filter = Filter(r, dr, gcutfilter, hfilter).filter
if orbital_free:
    vbar *= tw_coeff
vbar = my_filter(vbar * r)
#
# Calculate matrix elements:
dK_lnn = []
dH_lnn = []
dO_lnn = []
#
for l, (e_n, u_n, s_n, q_n) in enumerate(zip(e_ln, u_ln,
                                             s_ln, q_ln)):
    A_nn = np.inner(s_n, q_n * dr)
    # Do a LU decomposition of A:
    nn = len(e_n)
    L_nn = np.identity(nn, float)
    U_nn = A_nn.copy()
    #
    # Keep all bound states normalized
    if sum([n > 0 for n in n_ln[l]]) <= 1:
        for i in range(nn):
            for j in range(i + 1, nn):
                L_nn[j, i] = 1.0 * U_nn[j, i] / U_nn[i, i]
                U_nn[j, :] -= U_nn[i, :] * L_nn[j, i]
    #
    dO_nn = (np.inner(u_n, u_n * dr) - np.inner(s_n, s_n * dr))
    #
    e_nn = np.zeros((nn, nn))
    e_nn.ravel()[::nn + 1] = e_n
    dH_nn = np.dot(dO_nn, e_nn) - A_nn
    #
    q_n[:] = np.dot(inv(np.transpose(U_nn)), q_n)
    s_n[:] = np.dot(inv(L_nn), s_n)
    u_n[:] = np.dot(inv(L_nn), u_n)
    #
    dO_nn = np.dot(np.dot(inv(L_nn), dO_nn),
                   inv(np.transpose(L_nn)))
    dH_nn = np.dot(np.dot(inv(L_nn), dH_nn),
                   inv(np.transpose(L_nn)))
    #
    ku_n = [ae_calc.kin(l, u, e) for u, e in zip(u_n, e_n)]
    ks_n = [ae_calc.kin(l, s) for s in s_n]
    dK_nn = 0.5 * (np.inner(u_n, ku_n * dr) - np.inner(s_n, ks_n * dr))
    dK_nn += np.transpose(dK_nn).copy()
    #
    dK_lnn.append(dK_nn)
    dO_lnn.append(dO_nn)
    dH_lnn.append(dH_nn)
    #
    for n, q in enumerate(q_n):
        q[:] = my_filter(q, l) * r**(l + 1)
    #
    A_nn = np.inner(s_n, q_n * dr)
    q_n[:] = np.dot(inv(np.transpose(A_nn)), q_n)
#
#self.vt = vt
#self.vbar = vbar
#
print("state    eigenvalue         norm")
print("--------------------------------")
for l, (n_n, f_n, e_n) in enumerate(zip(n_ln, f_ln, e_ln)):
    for n in range(len(e_n)):
        if n_n[n] > 0:
            f = "(%d)" % f_n[n]
            print("%d%s%-4s: %12.6f %12.6f" % (
                n_n[n], "spdf"[l], f, e_n[n],
                np.dot(s_ln[l][n]**2, dr)))
        else:
            print("*%s    : %12.6f" % ("spdf"[l], e_n[n]))
print("--------------------------------")

# LOG DERIV calculation is skipped ....

# Writing data skipped ....



# Test for ghost states:
for h in [0.05]:
    check_diagonalize(r, h, N, beta, e_ln, n_ln, q_ln, emax, vt, lmax, dH_lnn, dO_lnn, ghost)



vn_j = []
vl_j = []
vf_j = []
ve_j = []
vu_j = []
vs_j = []
vq_j = []
j_ln = [[0 for f in f_n] for f_n in f_ln]
j = 0
for l, n_n in enumerate(n_ln):
    for n, nn in enumerate(n_n):
        if nn > 0:
            vf_j.append(f_ln[l][n])
            vn_j.append(nn)
            vl_j.append(l)
            ve_j.append(e_ln[l][n])
            vu_j.append(u_ln[l][n])
            vs_j.append(s_ln[l][n])
            vq_j.append(q_ln[l][n])
            j_ln[l][n] = j
            j += 1
for l, n_n in enumerate(n_ln):
    for n, nn in enumerate(n_n):
        if nn < 0:
            vf_j.append(0)
            vn_j.append(nn)
            vl_j.append(l)
            ve_j.append(e_ln[l][n])
            vu_j.append(u_ln[l][n])
            vs_j.append(s_ln[l][n])
            vq_j.append(q_ln[l][n])
            j_ln[l][n] = j
            j += 1
nj = j

dK_jj = np.zeros((nj, nj))
for l, j_n in enumerate(j_ln):
    for n1, j1 in enumerate(j_n):
        for n2, j2 in enumerate(j_n):
            dK_jj[j1, j2] = dK_lnn[l][n1, n2]
            if orbital_free:
                dK_jj[j1, j2] *= tw_coeff

X_gamma = yukawa_gamma
#if exx:
#    X_p = constructX(self)
#    if yukawa_gamma is not None and yukawa_gamma > 0:
#        X_pg = constructX(self, yukawa_gamma)
#    else:
#        X_pg = None
#    ExxC = atomic_exact_exchange(self, "core-core")
#else:
#    X_p = None
#    X_pg = None
#    ExxC = None

X_p = None
X_pg = None
ExxC = None



sqrt4pi = sqrt(4 * pi)
# This is the returned variable
setup = SetupData(symbol, ae_calc.xc.name, name, readxml=False, generator_version=1)

setup.l_j = vl_j
setup.l_orb_J = vl_j
setup.n_j = vn_j
setup.f_j = vf_j
setup.eps_j = ve_j
setup.rcut_j = [rcut_l[l] for l in vl_j]

setup.nc_g = nc * sqrt4pi
setup.nct_g = nct * sqrt4pi
setup.nvt_g = (nt - nct) * sqrt4pi
setup.e_kinetic_core = Ekincore
setup.vbar_g = vbar * sqrt4pi
setup.tauc_g = tauc * sqrt4pi
setup.tauct_g = tauct * sqrt4pi
setup.extra_xc_data = {} # extra_xc_data
setup.Z = Z
setup.Nc = Nc
setup.Nv = Nv
setup.e_kinetic = ae_calc.Ekin
setup.e_xc = ae_calc.Exc
setup.e_electrostatic = ae_calc.e_coulomb
setup.e_total = ae_calc.e_coulomb + ae_calc.Exc + ae_calc.Ekin

setup.rgd = ae_calc.rgd

setup.shape_function = {"type": "gauss",
                        "rc": rcutcomp / sqrt(gamma)}
setup.e_kin_jj = dK_jj
setup.ExxC = ExxC
setup.phi_jg = divide_all_by_r(r, vu_j, vl_j)
setup.phit_jg = divide_all_by_r(r, vs_j, vl_j)
setup.pt_jg = divide_all_by_r(r, vq_j, vl_j)
setup.X_p = X_p
setup.X_pg = X_pg
setup.X_gamma = X_gamma

# jcorehole stuffs is removed

if ghost and not orbital_free:
    # In orbital_free we are not interested in ghosts
    raise RuntimeError("Ghost!")

if scalarrel:
    reltype = "scalar-relativistic"
else:
    reltype = "non-relativistic"

attrs = [("type", reltype), ("name", "gpaw-%s" % version)]
data = "Frozen core: " + (core or "none")

setup.generatorattrs = attrs
setup.generatordata = data
setup.orbital_free = orbital_free
setup.version = "0.6"

id_j = []
for l, n in zip(vl_j, vn_j):
    if n > 0:
        my_id = "%s-%d%s" % (symbol, n, "spdf"[l])
    else:
        my_id = "%s-%s%d" % (symbol, "spdf"[l], -n)
    id_j.append(my_id)
setup.id_j = id_j

# if write_xml:
#    setup.write_xml()

# return setup


