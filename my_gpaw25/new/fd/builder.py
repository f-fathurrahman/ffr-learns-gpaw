from __future__ import annotations

from my_gpaw25.core import UGArray, UGDesc
from my_gpaw25.new.builder import create_uniform_grid
from my_gpaw25.new.fd.hamiltonian import FDHamiltonian
from my_gpaw25.new.fd.pot_calc import FDPotentialCalculator
from my_gpaw25.new.gpw import as_double_precision
from my_gpaw25.new.poisson import PoissonSolver, PoissonSolverWrapper
from my_gpaw25.new.pwfd.builder import PWFDDFTComponentsBuilder
from my_gpaw25.poisson import PoissonSolver as make_poisson_solver


class FDDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm,
                 nn=3,
                 interpolation=3):
        super().__init__(atoms,
                         params,
                         comm=comm)
        assert not self.soc
        self.kin_stencil_range = nn
        self.interpolation_stencil_range = interpolation

        self._nct_aR = None
        self._tauct_aR = None

        self.electrostatic_potential_desc = self.fine_grid
        self.interpolation_desc = self.fine_grid

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            self.mode,
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation='not fft',
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> UGDesc:
        return self.grid.new(dtype=self.dtype)

    def get_pseudo_core_densities(self):
        if self._nct_aR is None:
            self._nct_aR = self.setups.create_pseudo_core_densities(
                self.grid, self.relpos_ac, atomdist=self.atomdist, xp=self.xp)
        return self._nct_aR

    def get_pseudo_core_ked(self):
        if self._tauct_aR is None:
            self._tauct_aR = self.setups.create_pseudo_core_ked(
                self.grid, self.relpos_ac, atomdist=self.atomdist)
        return self._tauct_aR

    def create_poisson_solver(self) -> PoissonSolver:
        solver = make_poisson_solver(**self.params.poissonsolver, xp=self.xp)
        solver.set_grid_descriptor(self.fine_grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self):
        poisson_solver = self.create_poisson_solver()
        return FDPotentialCalculator(
            self.grid, self.fine_grid, self.setups, self.xc, poisson_solver,
            relpos_ac=self.relpos_ac, atomdist=self.atomdist,
            interpolation_stencil_range=self.interpolation_stencil_range,
            xp=self.xp)

    def create_hamiltonian_operator(self, blocksize=10):
        return FDHamiltonian(self.wf_desc, self.kin_stencil_range, blocksize,
                             xp=self.xp)

    def convert_wave_functions_from_uniform_grid(self,
                                                 C_nM,
                                                 basis_set,
                                                 kpt_c,
                                                 q):
        grid = self.grid.new(kpt=kpt_c, dtype=self.dtype)
        psit_nR = grid.zeros(self.nbands, self.communicators['b'])
        mynbands = len(C_nM.data)
        basis_set.lcao_to_grid(C_nM.data, psit_nR.data[:mynbands], q)
        return psit_nR.to_xp(self.xp)

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' in reader.wave_functions:
            name = 'coefficients'
        elif 'values' in reader.wave_functions:
            name = 'values'  # old name
        else:
            return ibzwfs

        singlep = reader.get('precision', 'double') == 'single'
        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file

        for wfs in ibzwfs:
            grid = self.wf_desc.new(kpt=wfs.kpt_c)
            index = (wfs.spin, wfs.k)
            data = reader.wave_functions.proxy(name, *index)
            data.scale = c
            if self.communicators['w'].size == 1 and not singlep:
                wfs.psit_nX = UGArray(grid, self.nbands, data=data)
            else:
                band_comm = self.communicators['b']
                wfs.psit_nX = UGArray(
                    grid, self.nbands,
                    comm=band_comm)
                if grid.comm.rank == 0:
                    mynbands = (self.nbands +
                                band_comm.size - 1) // band_comm.size
                    n1 = min(band_comm.rank * mynbands, self.nbands)
                    n2 = min((band_comm.rank + 1) * mynbands, self.nbands)
                    assert wfs.psit_nX.mydims[0] == n2 - n1
                    data = data[n1:n2]  # read from file

                if singlep:
                    wfs.psit_nX.scatter_from(as_double_precision(data))
                else:
                    wfs.psit_nX.scatter_from(data)

        return ibzwfs
