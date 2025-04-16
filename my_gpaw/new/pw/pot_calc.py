import numpy as np
from my_gpaw.core import PlaneWaves
from my_gpaw.gpu import cupy as cp
from my_gpaw.gpu import is_hip
from my_gpaw.mpi import broadcast_float
from my_gpaw.new import zip
from my_gpaw.new.pot_calc import PotentialCalculator
from my_gpaw.new.pw.stress import calculate_stress
from my_gpaw.setup import Setups


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups: Setups,
                 xc,
                 poisson_solver,
                 nct_ag,
                 nct_R,
                 soc=False,
                 xp=np):
        fracpos_ac = nct_ag.fracpos_ac
        atomdist = nct_ag.atomdist
        self.xp = xp
        super().__init__(xc, poisson_solver, setups, nct_R, fracpos_ac, soc)

        self.nct_ag = nct_ag
        self.vbar_ag = setups.create_local_potentials(
            pw, fracpos_ac, atomdist, xp)
        self.ghat_aLh = setups.create_compensation_charges(
            fine_pw, fracpos_ac, atomdist, xp)

        self.pw = pw
        self.fine_pw = fine_pw
        self.pw0 = pw.new(comm=None)  # not distributed

        self.h_g, self.g_r = fine_pw.map_indices(self.pw0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

        # There is a bug in HIP cupyx.scipy.rfftn:
        self.xp0 = cp if xp is cp and not is_hip else np

        self.fftplan = grid.fft_plans(xp=self.xp0)
        self.fftplan2 = fine_grid.fft_plans(xp=self.xp0)

        self.grid = grid
        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros(xp=xp)
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()

        self._nt_g = None
        self._vt_g = None

        self.e_stress = np.nan

    def calculate_charges(self, vHt_h):
        return self.ghat_aLh.integrate(vHt_h)

    def calculate_non_selfconsistent_exc(self, nt_sR, xc):
        nt_sr, _, _ = self._interpolate_density(nt_sR)
        vxct_sr = nt_sr.desc.empty(nt_sr.dims)
        e_xc = xc.calculate(nt_sr, vxct_sr)
        return e_xc

    def _interpolate_density(self, nt_sR):
        nt_sr = self.fine_grid.empty(nt_sR.dims, xp=self.xp0)
        pw = self.vbar_g.desc

        if pw.comm.rank == 0:
            indices = self.xp0.asarray(self.pw0.indices(self.fftplan.shape))
            nt0_g = self.pw0.zeros(xp=self.xp)
        else:
            nt0_g = None

        ndensities = nt_sR.dims[0] % 3
        for spin, (nt_R, nt_r) in enumerate(zip(nt_sR, nt_sr)):
            nt_R.to_xp(self.xp0).interpolate(
                self.fftplan, self.fftplan2, out=nt_r)
            if spin < ndensities and pw.comm.rank == 0:
                nt0_g.data += self.xp.asarray(
                    self.fftplan.tmp_Q.ravel()[indices])

        return nt_sr.to_xp(self.xp), pw, nt0_g

    def calculate_pseudo_potential(self, density, vHt_h):
        nt_sr, pw, nt0_g = self._interpolate_density(density.nt_sR)

        vxct_sr = nt_sr.desc.empty(density.nt_sR.dims, xp=self.xp)
        e_xc = self.xc.calculate(nt_sr, vxct_sr)

        if pw.comm.rank == 0:
            nt0_g.data *= 1 / np.prod(density.nt_sR.desc.size_c)
            e_zero = self.vbar0_g.integrate(nt0_g)
        else:
            e_zero = 0.0
        e_zero = broadcast_float(float(e_zero), pw.comm)

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros(xp=self.xp)

        charge_h = vHt_h.desc.zeros(xp=self.xp)
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)

        if pw.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pw.comm.send(nt0_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pw.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        # background charge ???

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pw.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pw.comm.receive(data, rank)
                    vt0_g.data[g] += data
            vt0_R = vt0_g.to_xp(self.xp0).ifft(
                plan=self.fftplan,
                grid=density.nt_sR.desc.new(comm=None)).to_xp(self.xp)
        else:
            pw.comm.send(vHt_h.data[self.h_g], 0)

        vt_sR = density.nt_sR.new()
        vt_sR[0].scatter_from(vt0_R if pw.comm.rank == 0 else None)
        if density.ndensities == 2:
            vt_sR.data[1] = vt_sR.data[0]
        vt_sR.data[density.ndensities:] = 0.0

        e_kinetic = self._restrict(vxct_sr, vt_sR, density)

        e_external = 0.0

        self.e_stress = e_coulomb + e_zero

        self._reset()

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_h

    def _restrict(self, vxct_sr, vt_sR, density=None):
        vtmp_R = vt_sR.desc.empty(xp=self.xp0)
        e_kinetic = 0.0
        for spin, (vt_R, vxct_r) in enumerate(zip(vt_sR, vxct_sr)):
            vxct_r.to_xp(self.xp0).fft_restrict(
                self.fftplan2, self.fftplan, out=vtmp_R)
            vt_R.data += vtmp_R.to_xp(self.xp).data
            if density:
                e_kinetic -= vt_R.integrate(density.nt_sR[spin])
                if spin < density.ndensities:
                    e_kinetic += vt_R.integrate(self.nct_R)
        return float(e_kinetic)

    def restrict(self, vt_sr):
        vt_sR = self.grid.empty(vt_sr.dims, xp=self.xp0)
        for vt_R, vt_r in zip(vt_sR, vt_sr):
            vt_r.to_xp(self.xp0).fft_restrict(
                self.fftplan2, self.fftplan, out=vt_R)
        return vt_sR

    def _move(self, fracpos_ac, atomdist, ndensities):
        self.ghat_aLh.move(fracpos_ac, atomdist)
        self.vbar_ag.move(fracpos_ac, atomdist)
        self.vbar_g.data[:] = 0.0
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()
        self.nct_ag.move(fracpos_ac, atomdist)
        self.nct_ag.to_uniform_grid(out=self.nct_R, scale=1.0 / ndensities)
        self._reset()

    def _reset(self):
        self._vt_g = None
        self._nt_g = None

    def _force_stress_helper(self, state):
        if self._vt_g is None:
            density = state.density
            potential = state.potential
            nt_R = density.nt_sR[0]
            vt_R = potential.vt_sR[0]
            if density.ndensities > 1:
                nt_R = nt_R.desc.empty(xp=self.xp)
                nt_R.data[:] = density.nt_sR.data[:density.ndensities].sum(
                    axis=0)
                vt_R = vt_R.desc.empty(xp=self.xp)
                vt_R.data[:] = (
                    potential.vt_sR.data[:density.ndensities].sum(axis=0) /
                    density.ndensities)
            self._vt_g = vt_R.to_xp(self.xp0).fft(self.fftplan,
                                                  pw=self.pw).to_xp(self.xp)
            self._nt_g = nt_R.to_xp(self.xp0).fft(self.fftplan,
                                                  pw=self.pw).to_xp(self.xp)
        return self._vt_g, self._nt_g

    def force_contributions(self, state):
        vt_g, nt_g = self._force_stress_helper(state)

        return (self.ghat_aLh.derivative(state.vHt_x),
                self.nct_ag.derivative(vt_g),
                self.vbar_ag.derivative(nt_g))

    def stress(self, state):
        vt_g, nt_g = self._force_stress_helper(state)
        return calculate_stress(self, state, vt_g, nt_g)
