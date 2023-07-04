import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.tddft import TDDFT, DipoleMomentWriter, photoabsorption_spectrum
from gpaw.tddft.abc import PML, LinearAbsorbingBoundary, P4AbsorbingBoundary


@pytest.mark.later
def test_tddft_td_na2(in_tmp_dir):
    """Sodium dimer, Na2."""
    d = 1.5
    atoms = Atoms(symbols='Na2',
                  positions=[(0, 0, d),
                             (0, 0, -d)],
                  pbc=False)

    # Calculate ground state for TDDFT

    # Larger box
    atoms.center(vacuum=6.0)
    # Larger grid spacing, LDA is ok
    gs_calc = GPAW(nbands=1, h=0.35, xc='LDA', setups={'Na': '1'})
    atoms.calc = gs_calc
    atoms.get_potential_energy()
    gs_calc.write('na2_gs.gpw', 'all')

    # 16 fs run with 8.0 attosec time step
    time_step = 8.0  # 8.0 as (1 as = 0.041341 autime)5D
    iters = 10     # 2000 x 8 as => 16 fs
    # Weak delta kick to z-direction
    kick = [0, 0, 1e-3]

    # TDDFT calculator
    td_calc = TDDFT('na2_gs.gpw')
    DipoleMomentWriter(td_calc, 'na2_dmz.dat')
    # Kick
    td_calc.absorption_kick(kick)
    # Propagate
    td_calc.propagate(time_step, iters)
    td_calc.write('na2_td.gpw', mode='all')
    # Linear absorption spectrum
    photoabsorption_spectrum('na2_dmz.dat', 'na2_spectrum_z.dat', width=0.3)

    iters = 3

    # test restart
    td_rest = TDDFT('na2_td.gpw')
    DipoleMomentWriter(td_rest, 'na2_dmz.dat')
    td_rest.propagate(time_step, iters)

    # test restart
    td_rest = TDDFT('na2_td.gpw', solver='BiCGStab')
    DipoleMomentWriter(td_rest, 'na2_dmz3.dat', force_new_file=True)
    td_rest.propagate(time_step, iters)

    # test absorbing boundary conditions

    # linear imaginary potential
    td_ipabs = TDDFT('na2_gs.gpw')
    ip_abc = LinearAbsorbingBoundary(5.0, 0.01, atoms.positions)
    td_ipabs.set_absorbing_boundary(ip_abc)
    DipoleMomentWriter(td_ipabs, 'na2_dmz4.dat')
    td_ipabs.propagate(time_step, iters)

    # 4th order polynomial (1-(x^2-1)^2) imaginary potential
    td_ip4abs = TDDFT('na2_gs.gpw')
    ip4_abc = P4AbsorbingBoundary(5.0, 0.03, atoms.positions, 3.0)
    td_ip4abs.set_absorbing_boundary(ip4_abc)
    DipoleMomentWriter(td_ip4abs, 'na2_dmz5.dat')
    td_ip4abs.propagate(time_step, iters)

    # perfectly matched layers
    td_pmlabs = TDDFT('na2_gs.gpw', solver='BiCGStab')
    pml_abc = PML(100.0, 0.1)
    td_pmlabs.set_absorbing_boundary(pml_abc)
    DipoleMomentWriter(td_pmlabs, 'na2_dmz6.dat')
    td_pmlabs.propagate(time_step, iters)
