import pytest
from my_gpaw25 import GPAW, FermiDirac
from my_gpaw25.mpi import world, size, rank
from my_gpaw25.lrtddft2 import LrTDDFT2
from my_gpaw25.lrtddft2.lr_communicators import LrCommunicators
from ase.atoms import Atoms


@pytest.mark.lrtddft
def test_lrtddft2_Al2(in_tmp_dir):
    debug = False
    restart_file = 'Al2_gs.gpw'

    d = 2.563
    atoms = Atoms('Al2', positions=((0, 0, 0),
                                    (0, 0, d)))
    atoms.center(4.0)
    calc = GPAW(mode='fd', h=0.24, eigensolver='cg', basis='dzp',
                occupations=FermiDirac(width=0.01),
                convergence={'eigenstates': 4.0e-5,
                             'density': 1.0e-2,
                             'bands': 'all'},
                nbands=20)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(restart_file, mode='all')

    # Try to run parallel over eh-pairs
    if size % 2 == 0:
        eh_size = 2
        domain_size = size // eh_size
    else:
        eh_size = 1
        domain_size = size

    lr_comms = LrCommunicators(world, domain_size, eh_size)

    calc = GPAW(restart_file,
                communicator=lr_comms.dd_comm)
    de = 3.0
    lr = LrTDDFT2('Al2_lri',
                  calc,
                  fxc='PBE',
                  max_energy_diff=de,
                  lr_communicators=lr_comms
                  )
    w, S, R, Sx, Sy, Sz = lr.get_transitions(max_energy=1e9, units='au')

    e0_1 = w[0]
    e1_1 = w[-1]
    # Continue with larger de
    de = 4.5
    lr = LrTDDFT2('Al2_lri',
                  calc,
                  fxc='PBE',
                  max_energy_diff=de,
                  lr_communicators=lr_comms)
    w, S, R, Sx, Sy, Sz = lr.get_transitions(max_energy=1e9, units='au')
    e0_2 = w[0]
    e1_2 = w[-1]
    # Continue with smaller de
    de = 2.5
    lr = LrTDDFT2('Al2_lri',
                  calc,
                  fxc='PBE',
                  max_energy_diff=de,
                  lr_communicators=lr_comms)
    w, S, R, Sx, Sy, Sz = lr.get_transitions(max_energy=1e9, units='au')
    e0_3 = w[0]
    e1_3 = w[-1]

    if debug and rank == 0:
        print(e0_1, e1_1)
        print(e0_2, e1_2)
        print(e0_3, e1_3)

    tol = 5.0e-8
    assert e0_1 == pytest.approx(e0_2, abs=tol)
    assert e0_1 == pytest.approx(e0_3, abs=tol)
    tol = 1.0e-3
    assert e0_1 == pytest.approx(0.00105074187176, abs=tol)
    assert e1_1 == pytest.approx(0.183188157301, abs=tol)
    assert e1_2 == pytest.approx(0.194973135812, abs=tol)
    assert e1_3 == pytest.approx(0.120681529342, abs=tol)
