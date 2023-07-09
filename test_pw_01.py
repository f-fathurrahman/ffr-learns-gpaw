import numpy as np
from ase.units import Bohr, Ha
from my_gpaw import GPAW, PW

A = 16*Bohr
cell_vectors = np.array([
    [A, 0.0, 0.0],
    [0.0, A, 0.0],
    [0.0, 0.0, A]
])

ecutwfc = 15*Ha
pw = PW(ecut=ecutwfc, cell=cell_vectors)

