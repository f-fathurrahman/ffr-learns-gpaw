# This is for creating atomic PAW setup

# There is also another similar function create_setups
# which is used for setting up DFT calculation

import numpy as np

import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

from my_gpaw25.setup import create_setup

atom_data = create_setup("Pd")

vbar = atom_data.vbar # of Spline type
r = atom_data.rgd.r_g
vbar_arr = np.zeros(r.shape)
for i in range(len(r)):
    vbar_arr[i] = vbar(r[i])

plt.figure(figsize=(4,2))
plt.plot(r, vbar_arr)
plt.xlim(0.0, 2.0);

# or access directly from atom_data.data
r_g = atom_data.data.rgd.r_g

vbar_g = atom_data.data.vbar_g

plt.figure(figsize=(4,2))
plt.plot(r_g, vbar_g)
plt.xlim(0, 5);

phi_jg = atom_data.data.phi_jg
phit_jg = atom_data.data.phit_jg

n_j = atom_data.n_j
l_j = atom_data.l_j
eps_j = atom_data.data.eps_j

plt.figure(figsize=(6,4))
iproj = 0
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 2.0); plt.ylim(-0.5, 0.5);

plt.figure(figsize=(6,4))
iproj = 1
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 5.0); plt.ylim(-1.5, 1.5);

plt.figure(figsize=(6,4))
iproj = 2
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 5.0); plt.ylim(-1.5, 1.5);

plt.figure(figsize=(6,4))
iproj = 3
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 5.0); plt.ylim(-1.5, 1.5);

plt.figure(figsize=(6,4))
iproj = 4
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 10.0); plt.ylim(-1.5, 1.5);

plt.figure(figsize=(6,4))
iproj = 5
plt.plot(r_g, phi_jg[iproj], label="AE")
plt.plot(r_g, phit_jg[iproj], label="PS")
plt.title(f"n={n_j[iproj]} l={l_j[iproj]} eps_j={eps_j[iproj]:.3f}")
plt.legend()
plt.xlim(0.0, 10.0); plt.ylim(-1.5, 1.5);

atom_data.rcut_j
