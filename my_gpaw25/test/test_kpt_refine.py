import pytest
from ase.lattice.hexagonal import Graphene
from ase.parallel import parprint

from my_gpaw25 import GPAW, PW, FermiDirac


def test_kpt_refine(gpaw_new):
    if gpaw_new:
        pytest.skip('Not implemented')
    system = Graphene(symbol='C',
                      latticeconstant={'a': 2.467710, 'c': 1.0},
                      size=(1, 1, 1))
    system.pbc = (1, 1, 0)
    system.center(axis=2, vacuum=4.0)

    kpt_refine = {
        'center': [1 / 3, 1 / 3, 0],
        'size': [3, 3, 1],
        'reduce_symmetry': False}
    # kpt_refine={"center":[[1./3,1./3,0.],[-1./3,-1./3,0.]], "size":[3,3,1],
    #             "reduce_symmetry":True}

    calc = GPAW(mode=PW(ecut=400),
                xc=dict(name='PBE', stencil=1),
                kpts={'size': [9, 9, 1], 'gamma': True},
                experimental={'kpt_refine': kpt_refine},
                occupations=FermiDirac(0.026))

    system.calc = calc
    energy = system.get_potential_energy()

    parprint('Energy', energy)

    assert abs(energy - -18.27) < 0.02
