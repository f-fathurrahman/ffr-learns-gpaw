from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode=PW(300),  # energy cutoff for plane wave basis (in eV)
            kpts={'size': (3, 3, 3), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='C_groundstate.txt')

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()  # determine all bands
calc.write('C_groundstate.gpw', 'all')  # write out wavefunctions


from gpaw.response.g0w0 import G0W0

gw = G0W0(calc='C_groundstate.gpw',
          nbands=30,  # number of bands for calculation of self-energy
          bands=(3, 5),          # VB and CB
          ecut=20.0,             # plane-wave cutoff for self-energy
          integrate_gamma='WS',  # Use supercell Wigner-Seitz truncation for W.
          filename='C-g0w0')
gw.calculate()



import pickle

results = pickle.load(open('C-g0w0_results_GW.pckl', 'rb'))
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2]

print('Direct bandgap of C:', direct_gap)