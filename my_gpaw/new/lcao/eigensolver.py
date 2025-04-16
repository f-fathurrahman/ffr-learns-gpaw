import numpy as np

from my_gpaw.new.eigensolver import Eigensolver
from my_gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from my_gpaw.new.lcao.wave_functions import LCAOWaveFunctions


class LCAOEigensolver(Eigensolver):
    def __init__(self, basis):
        self.basis = basis

    def iterate(self, state, hamiltonian) -> float:
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            state)

        for wfs in state.ibzwfs:
            self.iterate1(wfs, matrix_calculator)
        return 0.0

    def iterate1(self,
                 wfs: LCAOWaveFunctions,
                 matrix_calculator: HamiltonianMatrixCalculator):
        H_MM = matrix_calculator.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        N = min(len(eig_M), wfs.nbands)
        wfs._eig_n = np.empty(wfs.nbands)
        wfs._eig_n[:N] = eig_M[:N]
        wfs.C_nM.data[:N] = H_MM.data.T[:N]

        # Make sure wfs.C_nM and (lazy) wfs.P_ani are in sync:
        wfs._P_ani = None
