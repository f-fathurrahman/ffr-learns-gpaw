import sys
sys.path.append("../")

from my_gpaw25.atom.configurations import parameters
from my_gpaw25.atom.generator import Generator

import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color": "gray"
})


xcname = "LDA"
symbol = "Pd"
par = parameters[symbol]

print("par = ", par)

filename = symbol + '.' + xcname + '.xml'
my_gen = Generator(symbol, xcname, scalarrel=True, nofiles=True)
my_gen.run(exx=True, logderiv=False, write_xml=False, **par)

r = my_gen.r
u_ln = my_gen.u_ln
s_ln = my_gen.s_ln
n_ln = my_gen.n_ln
lmax = my_gen.lmax
e_ln = my_gen.e_ln
f_ln = my_gen.f_ln

for l in range(my_gen.lmax+1):
    Nl = len(s_ln[l])
    for i in range(Nl):
        plt.clf()
        plt.plot(r, u_ln[l][i], label=f"u_{l}{i}")
        plt.plot(r, s_ln[l][i], label=f"s_{l}{i}")
        plt.xlim(0.0, 10.0)
        plt.title(f"n={n_ln[l][i]} l={l} e={e_ln[l][i]:.2f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"IMG_ae_ps_partialwave_{l}_{i}.png", dpi=150)
