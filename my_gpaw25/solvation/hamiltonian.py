from my_gpaw25.hamiltonian import RealSpaceHamiltonian
from my_gpaw25.solvation.poisson import WeightedFDPoissonSolver
from my_gpaw25.fd_operators import Gradient
from my_gpaw25.io.logger import indent
from ase.units import Ha
import numpy as np


class SolvationRealSpaceHamiltonian(RealSpaceHamiltonian):
    """Realspace Hamiltonian with continuum solvent model.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(
        self,
        # solvation related arguments:
        cavity, dielectric, interactions,
        # RealSpaceHamiltonian arguments:
        gd, finegd, nspins, collinear, setups, timer, xc, world,
        redistributor, vext=None, psolver=None,
        stencil=3):
        """Constructor of SolvationRealSpaceHamiltonian class.

        Additional arguments not present in RealSpaceHamiltonian:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)
        if psolver is None:
            psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        self.gradient = None
        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, collinear, setups, timer, xc, world,
            vext=vext, psolver=psolver,
            stencil=stencil, redistributor=redistributor)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None
        self.e_el_free = None
        self.e_el_extrapolated = None

    def __str__(self):
        s = RealSpaceHamiltonian.__str__(self) + '\n'
        s += '  Solvation:\n'
        components = [self.cavity, self.dielectric] + self.interactions
        s += ''.join([indent(str(c), 2) for c in components])
        return s

    def estimate_memory(self, mem):
        RealSpaceHamiltonian.estimate_memory(self, mem)
        solvation = mem.subnode('Solvation')
        for name, obj in [
            ('Cavity', self.cavity),
            ('Dielectric', self.dielectric),
        ] + [('Interaction: ' + ia.subscript, ia) for ia in self.interactions]:
            obj.estimate_memory(solvation.subnode(name))

    def update_atoms(self, atoms, log):
        self.new_atoms = atoms.copy()
        log('Solvation position-dependent initialization:')
        self.cavity.update_atoms(atoms, log)
        for ia in self.interactions:
            ia.update_atoms(atoms, log)

    def initialize(self):
        self.gradient = [
            Gradient(self.finegd, i, 1.0, self.poisson.nn) for i in (0, 1, 2)
        ]
        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)

    def update(self, density, wfs=None, kin_en_using_band=True):
        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            self.timer.start('Initialize Hamiltonian')
            self.initialize()
            self.timer.stop('Initialize Hamiltonian')

        cavity_changed = self.cavity.update(self.new_atoms, density)
        if cavity_changed:
            self.cavity.update_vol_surf()
            self.dielectric.update(self.cavity)

        # e_coulomb, Ebar, Eext, Exc =
        finegd_energies = self.update_pseudo_potential(density)
        self.finegd.comm.sum(finegd_energies)
        ia_changed = [
            ia.update(
                self.new_atoms,
                density,
                self.cavity if cavity_changed else None)
            for ia in self.interactions]
        if np.any(ia_changed):
            self.vt_ia_g.fill(.0)
            for ia in self.interactions:
                if ia.depends_on_el_density:
                    self.vt_ia_g += ia.delta_E_delta_n_g
                if self.cavity.depends_on_el_density:
                    self.vt_ia_g += (ia.delta_E_delta_g_g *
                                     self.cavity.del_g_del_n_g)
        if len(self.interactions) > 0:
            for vt_g in self.vt_sg:
                vt_g += self.vt_ia_g
        Eias = np.array([ia.E for ia in self.interactions])

        Ekin1 = self.gd.comm.sum_scalar(self.calculate_kinetic_energy(density))
        W_aL = self.calculate_atomic_hamiltonians(density)
        atomic_energies = self.update_corrections(density, W_aL)
        self.world.sum(atomic_energies)

        energies = atomic_energies
        energies[1:] += finegd_energies
        energies[0] += Ekin1

        if not kin_en_using_band:
            assert wfs is not None
            with self.timer('New Kinetic Energy'):
                energies[0] = \
                    self.calculate_kinetic_energy_directly(density,
                                                           wfs)

        (self.e_kinetic0, self.e_coulomb, self.e_zero,
         self.e_external, self.e_xc) = energies

        self.finegd.comm.sum(Eias)

        self.cavity.communicate_vol_surf(self.world)
        for E, ia in zip(Eias, self.interactions):
            setattr(self, 'e_' + ia.subscript, E)

        self.new_atoms = None
        self.timer.stop('Hamiltonian')

    def update_pseudo_potential(self, density):
        ret = RealSpaceHamiltonian.update_pseudo_potential(self, density)
        if not self.cavity.depends_on_el_density:
            return ret
        del_g_del_n_g = self.cavity.del_g_del_n_g
        # XXX optimize numerics
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        Veps = -1. / (8. * np.pi) * del_eps_del_g_g * del_g_del_n_g
        Veps *= self.grad_squared(self.vHt_g)
        for vt_g in self.vt_sg:
            vt_g += Veps
        return ret

    def calculate_forces(self, dens, F_av):
        # XXX reorganize
        self.el_force_correction(dens, F_av)
        for ia in self.interactions:
            if self.cavity.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            ia.delta_E_delta_g_g * del_g_del_r_vg[v],
                            global_integral=False)
            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vg = ia.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            del_E_del_r_vg[v],
                            global_integral=False)
        return RealSpaceHamiltonian.calculate_forces(self, dens, F_av)

    def el_force_correction(self, dens, F_av):
        if not self.cavity.depends_on_atomic_positions:
            return
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        fixed = 1 / (8 * np.pi) * del_eps_del_g_g * \
            self.grad_squared(self.vHt_g)  # XXX grad_vHt_g inexact in bmgs
        for a, F_v in enumerate(F_av):
            del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
            for v in (0, 1, 2):
                F_v[v] += self.finegd.integrate(
                    fixed * del_g_del_r_vg[v],
                    global_integral=False)

    def get_energy(self, e_entropy, wfs, kin_en_using_band=True, e_sic=None):
        RealSpaceHamiltonian.get_energy(self, e_entropy, wfs,
                                        kin_en_using_band, e_sic)
        # The total energy calculated by the parent class includes the
        # solvent electrostatic contributions but not the interaction
        # energies. We add those here and store the electrostatic energies.
        self.e_el_free = self.e_total_free
        self.e_el_extrapolated = self.e_total_extrapolated
        for ia in self.interactions:
            self.e_total_free += getattr(self, 'e_' + ia.subscript)
        self.e_total_extrapolated = (self.e_total_free +
                                     wfs.occupations.extrapolate_factor *
                                     e_entropy)
        return self.e_total_free

    def grad_squared(self, x):
        # XXX ugly
        gs = np.empty_like(x)
        tmp = np.empty_like(x)
        self.gradient[0].apply(x, gs)
        np.square(gs, gs)
        self.gradient[1].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        self.gradient[2].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        return gs

    def summary(self, wfs, log):
        # This is almost duplicate code to gpaw/hamiltonian's
        # Hamiltonian.summary, but with the cavity and interactions added.

        log('Energy contributions relative to reference atoms:',
            f'(reference = {self.setups.Eref * Ha:.6f})\n')

        energies = [('Kinetic:      ', self.e_kinetic),
                    ('Potential:    ', self.e_coulomb),
                    ('External:     ', self.e_external),
                    ('XC:           ', self.e_xc),
                    ('Entropy (-ST):', self.e_entropy),
                    ('Local:        ', self.e_zero)]

        if len(self.interactions) > 0:
            energies += [('Interactions', None)]
            for ia in self.interactions:
                energies += [(f' {ia.subscript:s}:',
                              getattr(self, 'e_' + ia.subscript))]

        for name, e in energies:
            if e is not None:
                log('%-14s %+11.6f' % (name, Ha * e))
            else:
                log('%-14s' % (name))

        log('--------------------------')
        log('Free energy:   %+11.6f' % (Ha * self.e_total_free))
        log('Extrapolated:  %+11.6f' % (Ha * self.e_total_extrapolated))
        log()
        self.xc.summary(log)

        try:
            workfunctions = self.get_workfunctions(wfs)
        except ValueError:
            pass
        else:
            log('Dipole-layer corrected work functions: {:.6f}, {:.6f} eV'
                .format(*np.array(workfunctions) * Ha))
            log()

        self.cavity.summary(log)
