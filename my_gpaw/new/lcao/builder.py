from functools import partial

import numpy as np
from my_gpaw.core.matrix import Matrix
from my_gpaw.lcao.tci import TCIExpansions
from my_gpaw.new import zip
from my_gpaw.new.fd.builder import FDDFTComponentsBuilder
from my_gpaw.new.ibzwfs import create_ibz_wave_functions as create_ibzwfs
from my_gpaw.new.lcao.eigensolver import LCAOEigensolver
from my_gpaw.new.lcao.forces import TCIDerivatives
from my_gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from my_gpaw.new.lcao.hybrids import HybridLCAOEigensolver, HybridXCFunctional
from my_gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from my_gpaw.utilities.timing import NullTimer


class LCAODFTComponentsBuilder(FDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 distribution=None):
        super().__init__(atoms, params)
        self.distribution = distribution
        self.basis = None

    def create_wf_description(self):
        raise NotImplementedError

    def create_xc_functional(self):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return HybridXCFunctional(self.params.xc)
        return super().create_xc_functional()

    def create_basis_set(self):
        self.basis = FDDFTComponentsBuilder.create_basis_set(self)
        return self.basis

    def create_hamiltonian_operator(self):
        return LCAOHamiltonian(self.basis)

    def create_eigensolver(self, hamiltonian):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return HybridLCAOEigensolver(self.basis,
                                         self.fracpos_ac,
                                         self.grid.cell_cv)
        return LCAOEigensolver(self.basis)

    def read_ibz_wave_functions(self, reader):
        c = 1
        if reader.version >= 0 and reader.version < 4:
            c = reader.bohr**1.5

        basis = self.create_basis_set()
        potential = self.create_potential_calculator()
        if 'coefficients' in reader.wave_functions:
            coefficients = reader.wave_functions.proxy('coefficients')
            coefficients.scale = c
        else:
            coefficients = None

        ibzwfs = self.create_ibz_wave_functions(basis, potential,
                                                coefficients=coefficients)

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)
        return ibzwfs

    def create_ibz_wave_functions(self,
                                  basis,
                                  potential,
                                  *,
                                  log=None,
                                  coefficients=None):
        ibzwfs, _ = create_lcao_ibzwfs(
            basis, potential,
            self.ibz, self.communicators, self.setups,
            self.fracpos_ac, self.grid, self.dtype,
            self.nbands, self.ncomponents, self.atomdist, self.nelectrons,
            coefficients)
        return ibzwfs


def create_lcao_ibzwfs(basis, potential,
                       ibz, communicators, setups,
                       fracpos_ac, grid, dtype,
                       nbands, ncomponents, atomdist, nelectrons,
                       coefficients=None):
    kpt_comm = communicators['k']
    band_comm = communicators['b']
    domain_comm = communicators['d']

    rank_k = ibz.ranks(kpt_comm)
    here_k = rank_k == kpt_comm.rank
    kpt_qc = ibz.kpt_kc[here_k]

    tciexpansions = TCIExpansions.new_from_setups(setups)
    # basis.set_matrix_distribution(self.ksl.Mstart, self.ksl.Mstop)
    manytci = tciexpansions.get_manytci_calculator(
        setups, grid._gd, fracpos_ac,
        kpt_qc, dtype, NullTimer())

    my_atom_indices = basis.my_atom_indices
    S_qMM, T_qMM = manytci.O_qMM_T_qMM(domain_comm,
                                       0, setups.nao,
                                       False)
    if dtype == complex:
        np.negative(S_qMM.imag, S_qMM.imag)
        np.negative(T_qMM.imag, T_qMM.imag)

    P_aqMi = manytci.P_aqMi(my_atom_indices)
    P_qaMi = [{a: P_aqMi[a][q] for a in my_atom_indices}
              for q in range(len(S_qMM))]

    for a, P_qMi in P_aqMi.items():
        dO_ii = setups[a].dO_ii
        for P_Mi, S_MM in zip(P_qMi, S_qMM):
            S_MM += P_Mi.conj() @ dO_ii @ P_Mi.T
    domain_comm.sum(S_qMM)

    # self.atomic_correction= self.atomic_correction_cls.new_from_wfs(self)
    # self.atomic_correction.add_overlap_correction(newS_qMM)

    nao = setups.nao

    tci_derivatives = TCIDerivatives(manytci, atomdist, nao)

    def create_wfs(spin, q, k, kpt_c, weight):
        C_nM = Matrix(nbands, 2 * nao if ncomponents == 4 else nao,
                      dtype,
                      dist=(band_comm, band_comm.size, 1))
        if coefficients is not None:
            C_nM.data[:] = coefficients.proxy(spin, k)
        return LCAOWaveFunctions(
            setups=setups,
            density_adder=partial(basis.construct_density, q=q),
            tci_derivatives=tci_derivatives,
            basis=basis,
            C_nM=C_nM,
            S_MM=Matrix(nao, nao, data=S_qMM[q],
                        dist=(band_comm, band_comm.size, 1)),
            T_MM=T_qMM[q],
            P_aMi=P_qaMi[q],
            kpt_c=kpt_c,
            fracpos_ac=fracpos_ac,
            atomdist=atomdist,
            domain_comm=domain_comm,
            spin=spin,
            q=q,
            k=k,
            weight=weight,
            ncomponents=ncomponents)

    ibzwfs = create_ibzwfs(ibz,
                           nelectrons,
                           ncomponents,
                           create_wfs,
                           kpt_comm)
    return ibzwfs, tciexpansions
