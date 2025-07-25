from math import pi

import numpy as np
from ase.units import Hartree

from my_gpaw25.xc.sic import SIC
from my_gpaw25.atom.generator import Generator
from my_gpaw25.atom.configurations import parameters
from my_gpaw25.utilities import hartree


class NSCFSIC:
    r"""
    Helper object for applying non-self-consistent self-interaction
    corrections.

    :param paw:
        Converged GPAW calculator
    :type paw: :class:`my_gpaw25.calculator.GPAW`
    :param \**sic_parameters:
        Keyword arguments used to initialize the :class:`my_gpaw25.xc.sic.SIC`
        object, defaults to :py:attr:`~sic_defaults`
    """
    # NOTE: I'm only guessing what this part of the code does. Pls get someone
    # more competent to review.

    def __init__(self, paw, **sic_parameters):
        self.paw = paw
        self.sic_parameters = dict(self.sic_defaults, **sic_parameters)

    def calculate(self):
        ESIC = 0
        xc = self.paw.hamiltonian.xc
        assert xc.type == 'LDA'

        # Calculate the contribution from the core orbitals
        for a in self.paw.density.D_asp:
            setup = self.paw.density.setups[a]
            # TODO: Use XC which has been used to calculate the actual
            # calculation.
            # TODO: Loop over setups, not atoms.
            print('Atom core SIC for ', setup.symbol)
            print('%10s%10s%10s' % ('E_xc[n_i]', 'E_Ha[n_i]', 'E_SIC'))
            g = Generator(setup.symbol, xcname='LDA', nofiles=True, txt=None)
            g.run(**parameters[setup.symbol])
            njcore = g.njcore
            for f, l, e, u in zip(g.f_j[:njcore], g.l_j[:njcore],
                                  g.e_j[:njcore], g.u_j[:njcore]):
                # Calculate orbital density
                # NOTE: It's spherically symmetrized!
                # n = np.dot(self.f_j,
                assert l == 0, ('Not tested for l>0 core states')
                na = np.where(abs(u) < 1e-160, 0, u)**2 / (4 * pi)
                na[1:] /= g.r[1:]**2
                na[0] = na[1]
                nb = np.zeros(g.N)
                v_sg = np.zeros((2, g.N))
                vHr = np.zeros(g.N)
                Exc = xc.calculate_spherical(g.rgd, np.array([na, nb]), v_sg)
                hartree(0, na * g.r * g.dr, g.r, vHr)
                EHa = 2 * pi * np.dot(vHr * na * g.r, g.dr)
                print('{:10.2f}{:10.2f}{:10.2f}'.format(Exc * Hartree,
                                                        EHa * Hartree,
                                                        -f * (EHa +
                                                              Exc) * Hartree))
                ESIC += -f * (EHa + Exc)

        sic = SIC(**self.sic_parameters)
        sic.initialize(self.paw.density, self.paw.hamiltonian, self.paw.wfs)
        sic.set_positions(self.paw.spos_ac)

        print('Valence electron sic ')
        print('%10s%10s%10s%10s%10s%10s' % ('spin', 'k-point', 'band',
                                            'E_xc[n_i]', 'E_Ha[n_i]', 'E_SIC'))
        assert len(self.paw.wfs.kpt_u) == 1, 'Not tested for bulk calculations'

        for s, spin in sic.spin_s.items():
            spin.initialize_orbitals()
            spin.update_optimal_states()
            spin.update_potentials()

            n = 0
            for xc, c in zip(spin.exc_m, spin.ecoulomb_m):
                print('%10i%10i%10i%10.2f%10.2f%10.2f' %
                      (s, 0, n, -xc * Hartree, -c * Hartree,
                       2 * (xc + c) * Hartree))
                n += 1

            ESIC += spin.esic

        print('Total correction for self-interaction energy:')
        print('%10.2f eV' % (ESIC * Hartree))
        print('New total energy:')
        total = (ESIC * Hartree + self.paw.get_potential_energy() +
                 self.paw.get_reference_energy())
        print('%10.2f eV' % total)
        return total

    sic_defaults = dict(finegrid=True,
                        coulomb_factor=1,
                        xc_factor=1)
