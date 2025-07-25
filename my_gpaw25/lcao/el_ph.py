import pickle
import numpy as np
from os.path import isfile

from ase.parallel import parprint
from ase.units import Bohr, Ha

from my_gpaw25.mpi import world
from my_gpaw25.utilities import unpack_hermitian
from my_gpaw25.lcao.projected_wannier import dots
from my_gpaw25.utilities.tools import tri2full
from my_gpaw25.lfc import LocalizedFunctionsCollection as LFC


"""This module is used to calculate the electron-phonon coupling matrix,
    expressed in terms of GPAW LCAO orbitals.

    This module is not maintained and possibly broken.
    Use gpaw/elph/* instead.
    """


class ElectronPhononCouplingMatrix:
    r"""Class for calculating the electron-phonon coupling matrix, defined
    by the electron phonon interaction.

    ::

                  __                   _____
                  \     l   cc        /  h             cc
        H      =   )   M   c   c     /------   ( b  + b   ),
         el-ph    /_    ij  i   j  \/   2 W       l    l
                    l,ij                     l

    where the electron phonon coupling matrix is given by::

            l           ___
            M   = < i | \ /  V   * v  |j>
             ij          'u   eff   l

    """

    def __init__(self, atoms, indices=None, name='v', delta=0.005, nfree=2,
                 derivativemethod='tci'):
        assert nfree in [2, 4]
        self.nfree = nfree
        self.delta = delta

        if indices is None:
            indices = range(len(self.atoms))
        self.calc = atoms.calc
        self.atoms = atoms
        self.indices = np.asarray(indices)
        self.name = name
        self.p0 = self.atoms.positions.copy()

        self.derivativemethod = derivativemethod
        if derivativemethod == 'grid':
            self.get_dP_aMix = get_grid_dP_aMix
        elif derivativemethod == 'grid2':
            self.get_dP_aMix = get_grid2_dP_aMix
        elif derivativemethod == 'tci':
            self.get_dP_aMix = get_tci_dP_aMix
        else:
            raise ValueError('derivativemethod must be grid, grid2, or tci')

    def run(self):
        if not isfile(self.name + '.eq.pckl'):

            self.calc.calculate(self.atoms)
            Vt_G = self.calc.get_effective_potential(pad=False)
            Vt_G = self.calc.wfs.gd.collect(Vt_G, broadcast=True) / Ha
            dH_asp = self.calc.hamiltonian.dH_asp
            setups = self.calc.wfs.setups
            nspins = self.calc.wfs.nspins
            gd_comm = self.calc.wfs.gd.comm

            alldH_asp = {}
            for a, setup in enumerate(setups):
                ni = setup.ni
                nii = ni * (ni + 1) // 2
                tmpdH_sp = np.zeros((nspins, nii))
                if a in dH_asp:
                    tmpdH_sp[:] = dH_asp[a]
                gd_comm.sum(tmpdH_sp)
                alldH_asp[a] = tmpdH_sp

            forces = self.atoms.get_forces()
            self.calc.write('eq.gpw')

            world.barrier()
            if world.rank == 0:
                vd = open(self.name + '.eq.pckl', 'wb')
                fd = open('vib.eq.pckl', 'wb')
                pickle.dump((Vt_G, alldH_asp), vd, 2)
                pickle.dump(forces, fd)
                vd.close()
                fd.close()
            world.barrier()

        p = self.atoms.positions.copy()
        for a in self.indices:
            for j in range(3):
                for sign in [-1, 1]:
                    for ndis in range(1, self.nfree / 2 + 1):
                        name = '.%d%s%s.pckl' % (a, 'xyz'[j],
                                                 ndis * ' +-'[sign])
                        if isfile(self.name + name):
                            continue
                        self.atoms.positions[a, j] = (p[a, j] +
                                                      sign * ndis * self.delta)
                        self.calc.calculate(self.atoms)
                        Vt_G = self.calc.get_effective_potential(pad=False)
                        Vt_G = self.calc.wfs.gd.collect(Vt_G,
                                                        broadcast=True) / Ha
                        dH_asp = self.calc.hamiltonian.dH_asp

                        alldH_asp = {}
                        for a2, setup in enumerate(setups):
                            ni = setup.ni
                            nii = ni * (ni + 1) // 2
                            tmpdH_sp = np.zeros((nspins, nii))
                            if a2 in dH_asp:
                                tmpdH_sp[:] = dH_asp[a2]
                            gd_comm.sum(tmpdH_sp)
                            alldH_asp[a2] = tmpdH_sp

                        forces = self.atoms.get_forces()
                        world.barrier()
                        if world.rank == 0:
                            vd = open(self.name + name, 'wb')
                            fd = open('vib' + name, 'wb')
                            pickle.dump((Vt_G, alldH_asp), vd)
                            pickle.dump(forces, fd)
                            vd.close()
                            fd.close()
                        world.barrier()
                        self.atoms.positions[a, j] = p[a, j]
        self.atoms.set_positions(p)

    def get_gradient(self):
        """Calculates gradient"""
        nx = len(self.indices) * 3
        veqt_G, dHeq_asp = pickle.load(open(self.name + '.eq.pckl', 'rb'))
        gpts = veqt_G.shape
        dvt_Gx = np.zeros(gpts + (nx, ))
        ddH_aspx = {}
        for a, dH_sp in dHeq_asp.items():
            ddH_aspx[a] = np.empty(dH_sp.shape + (nx,))

        x = 0
        for a in self.indices:
            for i in range(3):
                name = '%s.%d%s' % (self.name, a, 'xyz'[i])
                vtm_G, dHm_asp = pickle.load(open(name + '-.pckl', 'rb'))
                vtp_G, dHp_asp = pickle.load(open(name + '+.pckl', 'rb'))

                if self.nfree == 4:
                    vtmm_G, dHmm_asp = pickle.load(
                        open(name + '--.pckl', 'rb'))
                    vtpp_G, dHpp_asp = pickle.load(
                        open(name + '++.pckl', 'rb'))
                    dvtdx_G = (-vtpp_G + 8.0 * vtp_G -
                               8 * vtm_G + vtmm_G) / (12 * self.delta / Bohr)
                    dvt_Gx[:, :, :, x] = dvtdx_G
                    for atom, ddH_spx in ddH_aspx.items():
                        ddH_aspx[atom][:, :, x] = (
                            -dHpp_asp[atom]
                            + 8.0 * dHp_asp[atom]
                            - 8.0 * dHm_asp[atom]
                            + dHmm_asp[atom]) / (12 * self.delta / Bohr)
                else:  # nfree = 2
                    dvtdx_G = (vtp_G - vtm_G) / (2 * self.delta / Bohr)
                    dvt_Gx[..., x] = dvtdx_G
                    for atom, ddH_spx in ddH_aspx.items():
                        ddH_aspx[atom][:, :, x] = (
                            dHp_asp[atom]
                            - dHm_asp[atom]) / (2 * self.delta / Bohr)
                x += 1
        return dvt_Gx, ddH_aspx

    def get_M(self, modes, q=0):
        r"""Calculate el-ph coupling matrix for given modes(s).

        XXX:
        kwarg "q=0" is an ugly hack for k-points.
        There shuold be loops over q!

        Note that modes must be given as a dictionary with mode
        frequencies in eV and corresponding mode vectors in units
        of 1/sqrt(amu), where amu = 1.6605402e-27 Kg is an atomic mass unit.
        In short frequencies and mode vectors must be given in ase units.

        ::

                  d                   d  ~
            < w | -- v | w' > = < w | -- v | w'>
                  dP                  dP

                               _
                              \        ~a     d   .       ~a
                            +  ) < w | p  >   -- /_\H   < p | w' >
                              /_        i     dP     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < w | -- p  >  /_\H    < p | w' >
                              /_       dP  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < w | p  >  /_\H    < -- p  | w' >
                              /_        i        ij    dP  j
                              a,ij

        """

        modes1 = modes.copy()
        # convert to atomic units
        amu = 1.6605402e-27  # atomic unit mass [Kg]
        me = 9.1093897e-31  # electron mass    [Kg]
        modes = {}
        for k in modes1.keys():
            modes[k / Ha] = modes1[k] / np.sqrt(amu / me)

        dvt_Gx, ddH_aspx = self.get_gradient()

        from my_gpaw25 import restart
        atoms, calc = restart('eq.gpw', txt=None)
        if calc.wfs.S_qMM is None:
            calc.initialize(atoms)
            calc.initialize_positions(atoms)

        wfs = calc.wfs
        nao = wfs.setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        spin = 0  # XXX

        M_lii = {}
        parprint('Starting gradient of pseudo part')
        for f, mode in modes.items():
            mo = []
            M_ii = np.zeros((nao, nao), dtype)
            for a in self.indices:
                mo.append(mode[a])
            mode = np.asarray(mo).flatten()
            dvtdP_G = np.dot(dvt_Gx, mode)
            bfs.calculate_potential_matrix(dvtdP_G, M_ii, q=q)
            tri2full(M_ii, 'L')
            M_lii[f] = M_ii
        parprint('Finished gradient of pseudo part')

        P_aqMi = calc.wfs.P_aqMi
        # Add the term
        #  _
        # \        ~a     d   .       ~a
        #  ) < w | p  >   -- /_\H   < p | w' >
        # /_        i     dP     ij    j
        # a,ij

        Ma_lii = {}
        for f, mode in modes.items():
            Ma_lii[f] = np.zeros_like(M_lii.values()[0])

        parprint('Starting gradient of dH^a part')
        for f, mode in modes.items():
            mo = []
            for a in self.indices:
                mo.append(mode[a])
            mode = np.asarray(mo).flatten()

            for a, ddH_spx in ddH_aspx.items():
                ddHdP_sp = np.dot(ddH_spx, mode)
                ddHdP_ii = unpack_hermitian(ddHdP_sp[spin])
                Ma_lii[f] += dots(P_aqMi[a][q], ddHdP_ii, P_aqMi[a][q].T)
        parprint('Finished gradient of dH^a part')

        parprint('Starting gradient of projectors part')
        dP_aMix = self.get_dP_aMix(calc.spos_ac, wfs, q)
        parprint('Finished gradient of projectors part')

        dH_asp = pickle.load(open('v.eq.pckl', 'rb'))[1]

        Mb_lii = {}
        for f, mode in modes.items():
            Mb_lii[f] = np.zeros_like(M_lii.values()[0])

        for f, mode in modes.items():
            for a, dP_Mix in dP_aMix.items():
                dPdP_Mi = np.dot(dP_Mix, mode[a])
                dH_ii = unpack_hermitian(dH_asp[a][spin])
                dPdP_MM = dots(dPdP_Mi, dH_ii, P_aqMi[a][q].T)
                Mb_lii[f] -= dPdP_MM + dPdP_MM.T
                # XXX The minus sign here is quite subtle.
                # It is related to how the derivative of projector
                # functions in GPAW is calculated.
                # More thorough explanations, anyone...?

        # Units of M_lii are Hartree/(Bohr * sqrt(m_e))
        for mode in M_lii.keys():
            M_lii[mode] += Ma_lii[mode] + Mb_lii[mode]

        # conversion to eV. The prefactor 1 / sqrt(hb^2 / 2 * hb * f)
        # has units Bohr * sqrt(me)
        M_lii_1 = M_lii.copy()
        M_lii = {}

        for f in M_lii_1.keys():
            M_lii[f * Ha] = M_lii_1[f] * Ha / np.sqrt(2 * f)

        return M_lii


#####################################################
# XXX grid and grid 2 sometimes gives random numbers,
# XXX sometimes even nan!
#####################################################

def get_grid_dP_aMix(spos_ac, wfs, q):  # XXXXXX q
    nao = wfs.setups.nao
    C_MM = np.identity(nao, dtype=wfs.dtype)
    # XXX In the future use the New Two-Center integrals
    # to evaluate this
    dP_aMix = {}
    for a, setup in enumerate(wfs.setups):
        ni = 0
        dP_Mix = np.zeros((nao, setup.ni, 3))
        pt = LFC(wfs.gd, [setup.pt_j],
                 wfs.kd.comm, dtype=wfs.dtype, forces=True)
        spos1_ac = [spos_ac[a]]
        pt.set_k_points(wfs.ibzk_qc)
        pt.set_positions(spos1_ac)
        for b, setup_b in enumerate(wfs.setups):
            nao = setup_b.nao
            phi_MG = wfs.gd.zeros(nao, wfs.dtype)
            phi_MG = wfs.gd.collect(phi_MG, broadcast=False)
            wfs.basis_functions.lcao_to_grid(C_MM[ni:ni + nao], phi_MG, q)
            dP_bMix = pt.dict(len(phi_MG), derivative=True)
            pt.derivative(phi_MG, dP_bMix, q=q)
            dP_Mix[ni:ni + nao] = dP_bMix[0]
            ni += nao
            parprint(f'projector grad. doing atoms ({a}, {b}) ')

        dP_aMix[a] = dP_Mix
    return dP_aMix


def get_grid2_dP_aMix(spos_ac, wfs, q, *args, **kwargs):  # XXXXXX q
    nao = wfs.setups.nao
    C_MM = np.identity(nao, dtype=wfs.dtype)
    bfs = wfs.basis_functions
    phi_MG = wfs.gd.zeros(nao, wfs.dtype)
    bfs.lcao_to_grid(C_MM, phi_MG, q)
    setups = wfs.setups
    pt = LFC(wfs.gd, [setup.pt_j for setup in setups],
             wfs.kd.comm, dtype=wfs.dtype, forces=True)
    pt.set_k_points(wfs.ibzk_qc)
    pt.set_positions(spos_ac)
    dP_aMix = pt.dict(len(phi_MG), derivative=True)
    pt.derivative(phi_MG, dP_aMix, q=q)
    return dP_aMix


def get_tci_dP_aMix(spos_ac, wfs, q, *args, **kwargs):
    # container for spline expansions of basis function-projector pairs
    # (note to self: remember to conjugate/negate because of that)
    from my_gpaw25.lcao.overlap import ManySiteDictionaryWrapper, \
        TwoCenterIntegralCalculator, NewTwoCenterIntegrals

    if not isinstance(wfs.tci, NewTwoCenterIntegrals):
        raise RuntimeError('Please remember --gpaw=usenewtci=True')

    dP_aqxMi = {}
    nao = wfs.setups.nao
    nq = len(wfs.ibzk_qc)
    for a, setup in enumerate(wfs.setups):
        dP_aqxMi[a] = np.zeros((nq, 3, nao, setup.ni), wfs.dtype)

    calc = TwoCenterIntegralCalculator(wfs.ibzk_qc, derivative=True)
    expansions = ManySiteDictionaryWrapper(wfs.tci.P_expansions, dP_aqxMi)
    calc.calculate(wfs.tci.atompairs, [expansions], [dP_aqxMi])

    dP_aMix = {}
    for a in dP_aqxMi:
        dP_aMix[a] = dP_aqxMi[a].transpose(0, 2, 3, 1).copy()[q]  # XXX q
    return dP_aMix
