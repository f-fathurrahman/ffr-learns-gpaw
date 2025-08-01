# This is for creating atomic PAW setup

# There is also another similar function create_setups
# which is used for setting up DFT calculation

import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})

from my_gpaw25.setup import create_setup

atom_data = create_setup("Pd")

vbar = atom_data.vbar
r = atom_data.rgd.r_g
vbar_arr = np.zeros(r.shape)
for i in range(len(r)):
    vbar_arr[i] = vbar(r[i])

plt.plot(r, vbar_arr)
plt.xlim(0.0, 2.0)

# or access directly from atom_data.data
r_g = atom_data.data.rgd.r_g

vbar_g = atom_data.data.vbar_g

phi_jg = atom_data.data.phi_jg
phit_jg = atom_data.data.phit_jg