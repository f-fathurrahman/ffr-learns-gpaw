"""Test the reading of wave functions as file references."""
import numpy as np
from my_gpaw25 import GPAW
from my_gpaw25.mpi import rank, world


def test_fileio_file_reference(in_tmp_dir, gpw_files):
    # load restart from gpw
    calc = GPAW(gpw_files['na3_fd_kp_restart'])
    wf0 = calc.get_pseudo_wave_function(2, 1, 1)

    # Now read with a single process
    comm = world.new_communicator(np.array((rank,)))
    calc = GPAW(gpw_files['na3_fd_kp_restart'], communicator=comm)
    wf1 = calc.get_pseudo_wave_function(2, 1, 1)

    # compare wf restarts match
    diff = np.abs(wf0 - wf1)
    assert np.all(diff < 1e-12)
