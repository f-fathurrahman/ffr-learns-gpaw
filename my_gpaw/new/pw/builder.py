from math import pi

from ase.units import Ha
from my_gpaw.core import PlaneWaves, UniformGrid
from my_gpaw.core.domain import Domain
from my_gpaw.core.matrix import Matrix
from my_gpaw.core.plane_waves import PlaneWaveExpansions
from my_gpaw.new import zip
from my_gpaw.new.builder import create_uniform_grid
from my_gpaw.new.pw.hamiltonian import PWHamiltonian, SpinorPWHamiltonian
from my_gpaw.new.pw.poisson import make_poisson_solver
from my_gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from my_gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder
from my_gpaw.new.spinors import SpinorWaveFunctionDescriptor
from my_gpaw.typing import Array1D


class PWDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    interpolation = 'fft'

    def __init__(self, atoms, params, ecut=340, qspiral=None):
        self.ecut = ecut / Ha
        super().__init__(atoms, params)

        self.qspiral_v = (None if qspiral is None else
                          qspiral @ self.grid.icell * (2 * pi))
        self._nct_ag = None

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            'pw',
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation='fft',
            ecut=self.ecut,
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> Domain:
        pw = PlaneWaves(ecut=self.ecut,
                        cell=self.grid.cell,
                        comm=self.grid.comm,
                        dtype=self.dtype)
        if self.ncomponents == 4:
            return SpinorWaveFunctionDescriptor(pw, qspiral_v=self.qspiral_v)
        return pw

    def create_xc_functional(self):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return ...
        return super().create_xc_functional()

    def get_pseudo_core_densities(self):
        if self._nct_ag is None:
            pw = PlaneWaves(ecut=2 * self.ecut,
                            cell=self.grid.cell,
                            comm=self.grid.comm)
            self._nct_ag = self.setups.create_pseudo_core_densities(
                pw, self.fracpos_ac, self.atomdist, xp=self.xp)
        return self._nct_ag

    def create_poisson_solver(self, fine_pw, params):
        return make_poisson_solver(fine_pw,
                                   self.fine_grid,
                                   self.atoms.pbc,
                                   self.params.charge,
                                   **params)

    def create_potential_calculator(self):
        nct_ag = self.get_pseudo_core_densities()
        pw = nct_ag.pw
        fine_pw = pw.new(ecut=8 * self.ecut)
        poisson_solver = self.create_poisson_solver(
            fine_pw,
            self.params.poissonsolver or {'strength': 1.0})
        return PlaneWavePotentialCalculator(self.grid,
                                            self.fine_grid,
                                            pw,
                                            fine_pw,
                                            self.setups,
                                            self.xc,
                                            poisson_solver,
                                            nct_ag, self.nct_R,
                                            self.soc,
                                            self.xp)

    def create_hamiltonian_operator(self, blocksize=10):
        if self.ncomponents < 4:
            return PWHamiltonian(self.grid, self.wf_desc, self.xp)
        return SpinorPWHamiltonian()

    def convert_wave_functions_from_uniform_grid(self,
                                                 C_nM: Matrix,
                                                 basis_set,
                                                 kpt_c,
                                                 q):
        # Replace this with code that goes directly from C_nM to
        # psit_nG via PWAtomCenteredFunctions.
        # XXX

        grid = self.grid.new(kpt=kpt_c, dtype=self.dtype)
        pw = self.wf_desc.new(kpt=kpt_c)
        psit_nG = pw.empty(self.nbands, self.communicators['b'])

        if self.dtype == complex:
            emikr_R = grid.eikr(-kpt_c)

        mynbands, M = C_nM.dist.shape

        if self.ncomponents < 4:
            psit_nR = grid.zeros(mynbands)
            basis_set.lcao_to_grid(C_nM.data, psit_nR.data, q)

            for psit_R, psit_G in zip(psit_nR, psit_nG):
                if self.dtype == complex:
                    psit_R.data *= emikr_R
                psit_R.fft(out=psit_G)
        else:
            psit_sR = grid.empty(2)
            C_nsM = C_nM.data.reshape((mynbands, 2, M // 2))
            for psit_sG, C_sM in zip(psit_nG, C_nsM):
                psit_sR.data[:] = 0.0
                basis_set.lcao_to_grid(C_sM, psit_sR.data, q)
                psit_sR.data *= emikr_R
                for psit_G, psit_R in zip(psit_sG, psit_sR):
                    psit_R.fft(out=psit_G)

        return psit_nG.to_xp(self.xp)

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' not in reader.wave_functions:
            return ibzwfs

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file
        elif reader.version < 4:
            c /= self.grid.size_c.prod()

        index_kG = reader.wave_functions.indices

        for wfs in ibzwfs:
            pw = self.wf_desc.new(kpt=wfs.kpt_c)
            if wfs.spin == 0:
                check_g_vector_ordering(self.grid, pw, index_kG[wfs.k])

            index = (wfs.spin, wfs.k) if self.ncomponents != 4 else (wfs.k,)
            data = reader.wave_functions.proxy('coefficients', *index)
            data.scale = c
            data.length_of_last_dimension = pw.shape[0]
            orig_shape = data.shape
            data.shape = (self.nbands, ) + pw.shape

            if self.communicators['w'].size == 1:
                wfs.psit_nX = PlaneWaveExpansions(pw, self.nbands,
                                                  data=data)
                data.shape = orig_shape
            else:
                band_comm = self.communicators['b']
                wfs.psit_nX = PlaneWaveExpansions(pw, self.nbands,
                                                  comm=band_comm)
                if pw.comm.rank == 0:
                    mynbands = (self.nbands +
                                band_comm.size - 1) // band_comm.size
                    n1 = min(band_comm.rank * mynbands, self.nbands)
                    n2 = min((band_comm.rank + 1) * mynbands, self.nbands)
                    assert wfs.psit_nX.mydims[0] == n2 - n1
                    data = data[n1:n2]  # read from file
                for psit_G, array in zip(wfs.psit_nX, data):
                    psit_G.scatter_from(array)

        return ibzwfs


def check_g_vector_ordering(grid: UniformGrid,
                            pw: PlaneWaves,
                            index_G: Array1D) -> None:
    size = tuple(grid.size)
    if pw.dtype == float:
        size = (size[0], size[1], size[2] // 2 + 1)
    index0_G = pw.indices(size)
    nG = len(index0_G)
    assert (index0_G == index_G[:nG]).all()
    assert (index_G[nG:] == -1).all()
