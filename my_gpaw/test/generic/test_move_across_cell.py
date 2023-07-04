import pytest
from ase.build import molecule

from gpaw import GPAW, Davidson, MixerSum

# Move atom infinitesimally across cell border and test that SCF loop is still
# well converged afterwards.  If it is /not/ well converged, then the code
# which compensates for discontinuity of phases is probably broken.


@pytest.mark.later
def test_generic_move_across_cell():
    def test(calc):
        atoms = molecule('H2O', vacuum=2.5)
        atoms.pbc = 1

        # Translate O to corner:
        atoms.positions -= atoms.positions[0, None, :]

        # Be sure that we are on the positive axis:

        atoms.calc = calc

        eps = 1e-12
        atoms.positions[0, :] = eps
        atoms.get_potential_energy()
        atoms.positions[0, 2] -= 2 * eps
        atoms.get_potential_energy()

        print(calc.scf.niter)

        # We should be within the convergence criterion.
        # It runs a minimum of three iterations:
        assert calc.scf.niter == 3

    def kwargs():
        # Make sure MixerSum works for spin-paired system also:
        return dict(xc='oldLDA', mixer=MixerSum(0.7), kpts=[1, 1, 2])

    test(GPAW(mode='lcao', basis='sz(dzp)', h=0.3))
    test(GPAW(mode='pw', eigensolver=Davidson(3),
              experimental={'reuse_wfs_method': 'paw'}, **kwargs()))
    test(GPAW(mode='fd', h=0.3,
              experimental={'reuse_wfs_method': 'lcao'}, **kwargs()))

    # pw + lcao extrapolation is currently broken (PWLFC lacks integrate2):
    # test(GPAW(mode='pw',
    #           experimental={'reuse_wfs_method': 'lcao'},
    #           **kwargs()))
