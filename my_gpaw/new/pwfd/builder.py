from types import SimpleNamespace

import numpy as np

from my_gpaw.new.builder import DFTComponentsBuilder
from my_gpaw.new.calculation import DFTState
from my_gpaw.new.ibzwfs import create_ibz_wave_functions as create_ibzwfs
from my_gpaw.new.lcao.eigensolver import LCAOEigensolver
from my_gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from my_gpaw.new.pwfd.davidson import Davidson
from my_gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def create_eigensolver(self, hamiltonian):
        eigsolv_params = self.params.eigensolver.copy()
        name = eigsolv_params.pop('name', 'dav')
        assert name == 'dav'
        return Davidson(
            self.nbands,
            self.wf_desc,
            self.communicators['b'],
            hamiltonian.create_preconditioner,
            converge_bands=self.params.convergence.get('bands', 'occupied'),
            **eigsolv_params)

    def read_ibz_wave_functions(self, reader):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = SimpleNamespace(
                comm=domain_comm,
                dims=(self.nbands,),
                desc=self.wf_desc.new(kpt=kpt_c),
                data=None,
                xp=np)
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,  # type: ignore
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents)

            return wfs

        ibzwfs = create_ibzwfs(self.ibz,
                               self.nelectrons,
                               self.ncomponents,
                               create_wfs,
                               self.communicators['k'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis, potential, *, log):
        from my_gpaw.new.lcao.builder import create_lcao_ibzwfs

        if self.params.random:
            return self.create_random_ibz_wave_functions(log)

        # sl_default = self.params.parallel['sl_default']
        # sl_lcao = self.params.parallel['sl_lcao'] or sl_default

        lcao_ibzwfs, _ = create_lcao_ibzwfs(
            basis, potential,
            self.ibz, self.communicators, self.setups,
            self.fracpos_ac, self.grid, self.dtype,
            self.nbands, self.ncomponents, self.atomdist, self.nelectrons)

        state = DFTState(lcao_ibzwfs, None, potential)
        hamiltonian = LCAOHamiltonian(basis)
        LCAOEigensolver(basis).iterate(state, hamiltonian)

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = lcao_ibzwfs.wfs_qs[q][spin]
            assert lcaowfs.spin == spin

            # Convert to PW-coefs in PW-mode:
            psit_nX = self.convert_wave_functions_from_uniform_grid(
                lcaowfs.C_nM, basis, kpt_c, q)
            eig_n = lcaowfs._eig_n

            nao = lcaowfs.C_nM.shape[1]
            if nao < self.nbands:
                psit_nX[nao:].randomize()
                eig_n[nao:] = np.inf

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents)
            wfs._eig_n = eig_n
            return wfs

        return create_ibzwfs(self.ibz, self.nelectrons, self.ncomponents,
                             create_wfs, self.communicators['k'])

    def create_random_ibz_wave_functions(self, log):
        log('Initializing wave functions with random numbers')

        def create_wfs(spin, q, k, kpt_c, weight):
            desc = self.wf_desc.new(kpt=kpt_c)
            psit_nX = desc.empty(
                dims=(self.nbands,),
                comm=self.communicators['b'],
                xp=self.xp)
            psit_nX.randomize()

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents)

            eig_n = self.xp.empty(self.nbands)
            eig_n[:] = np.inf
            wfs._eig_n = eig_n
            return wfs

        return create_ibzwfs(self.ibz, self.nelectrons, self.ncomponents,
                             create_wfs, self.communicators['k'])
