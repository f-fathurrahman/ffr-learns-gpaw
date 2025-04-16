from typing import Callable

import _gpaw
import numpy as np
from my_gpaw.core.plane_waves import PlaneWaveExpansions
from my_gpaw.core.uniform_grid import UniformGridFunctions
from my_gpaw.gpu import cupy as cp
from my_gpaw.new import zip
from my_gpaw.new.hamiltonian import Hamiltonian


class PWHamiltonian(Hamiltonian):
    def __init__(self, grid, pw, xp):
        self.plan = grid.new(dtype=pw.dtype).fft_plans(xp=xp)
        self.pw_cache = {}

    def apply(self,
              vt_sR: UniformGridFunctions,
              psit_nG: PlaneWaveExpansions,
              out: PlaneWaveExpansions,
              spin: int) -> PlaneWaveExpansions:
        out_nG = out
        vt_R = vt_sR[spin].gather(broadcast=True)
        xp = psit_nG.xp
        grid = vt_R.desc.new(comm=None, dtype=psit_nG.desc.dtype)
        tmp_R = grid.empty(xp=xp)
        pw = psit_nG.desc
        if pw.comm.size == 1:
            pw_local = pw
        else:
            key = tuple(pw.kpt_c)
            pw_local = self.pw_cache.get(key)
            if pw_local is None:
                pw_local = pw.new(comm=None)
                self.pw_cache[key] = pw_local
        psit_G = pw_local.empty(xp=xp)
        e_kin_G = xp.asarray(psit_G.desc.ekin_G)
        domain_comm = psit_nG.desc.comm
        mynbands = psit_nG.mydims[0]
        for n1 in range(0, mynbands, domain_comm.size):
            n2 = min(n1 + domain_comm.size, mynbands)
            psit_nG[n1:n2].gather_all(psit_G)
            psit_G.ifft(out=tmp_R)
            tmp_R.data *= vt_R.data
            vtpsit_G = tmp_R.fft(pw=psit_G.desc)
            psit_G.data *= e_kin_G
            vtpsit_G.data += psit_G.data
            out_nG[n1:n2].scatter_from_all(vtpsit_G)
        return out_nG

    def create_preconditioner(self,
                              blocksize: int
                              ) -> Callable[[PlaneWaveExpansions,
                                             PlaneWaveExpansions,
                                             PlaneWaveExpansions], None]:
        return precondition


def precondition(psit_nG: PlaneWaveExpansions,
                 residual_nG: PlaneWaveExpansions,
                 out: PlaneWaveExpansions) -> None:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """

    xp = psit_nG.xp
    G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
    ekin_n = psit_nG.norm2('kinetic')

    if xp is np:
        for r_G, o_G, ekin in zip(residual_nG.data,
                                  out.data,
                                  ekin_n):
            _gpaw.pw_precond(G2_G, r_G, ekin, o_G)
        return

    out.data[:] = gpu_prec(ekin_n[:, np.newaxis],
                           G2_G[np.newaxis],
                           residual_nG.data)


@cp.fuse()
def gpu_prec(ekin, G2, residual):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * residual


def spinor_precondition(psit_nsG, residual_nsG, out):
    G2_G = psit_nsG.desc.ekin_G * 2
    for r_sG, o_sG, ekin in zip(residual_nsG.data,
                                out.data,
                                psit_nsG.norm2('kinetic').sum(1)):
        for r_G, o_G in zip(r_sG, o_sG):
            _gpaw.pw_precond(G2_G, r_G, ekin, o_G)


class SpinorPWHamiltonian(Hamiltonian):
    def apply(self,
              vt_xR: UniformGridFunctions,
              psit_nsG: PlaneWaveExpansions,
              out: PlaneWaveExpansions,
              spin: int):
        out_nsG = out
        pw = psit_nsG.desc

        if pw.qspiral_v is None:
            np.multiply(pw.ekin_G, psit_nsG.data, out_nsG.data)
        else:
            for s, sign in enumerate([1, -1]):
                ekin_G = 0.5 * ((pw.G_plus_k_Gv +
                                 0.5 * sign * pw.qspiral_v)**2).sum(1)
                np.multiply(ekin_G, psit_nsG.data[:, s], out_nsG.data[:, s])

        grid = vt_xR.desc.new(dtype=complex)

        v, x, y, z = vt_xR.data
        iy = y * 1j

        f_sR = grid.empty(2)
        g_R = grid.empty()

        for p_sG, o_sG in zip(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=pw).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=pw).data

        return out_nsG

    def create_preconditioner(self, blocksize):
        return spinor_precondition
