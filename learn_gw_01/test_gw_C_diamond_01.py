from my_g0w0 import G0W0

gw = G0W0(calc="C_groundstate.gpw",
          nbands=30,  # number of bands for calculation of self-energy
          bands=(3, 5),          # VB and CB
          ecut=20.0,             # plane-wave cutoff for self-energy
          integrate_gamma="WS",  # Use supercell Wigner-Seitz truncation for W.
          filename="LOG_g0w0")

#gw.calculate()
from debug_g0w0_calculate import g0w0_calculate
g0w0_calculate(gw)


import pickle
results = pickle.load(open("LOG_g0w0_results_GW.pckl", "rb"))
direct_gap = results["qp"][0, 0, -1] - results["qp"][0, 0, -2]
print("Direct bandgap of C:", direct_gap)