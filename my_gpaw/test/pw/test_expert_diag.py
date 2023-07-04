import pytest
import numpy as np
from ase.build import bulk

from gpaw import GPAW, PW
from gpaw.mpi import world

# This test is asserting that the expert diagonalization
# routine gives the same result as the non-expert version
# in terms of eigenvalues and wavefunctions


@pytest.mark.later
def test_pw_expert_diag(in_tmp_dir, scalapack):
    wfs_e = []
    for i, expert in enumerate([True, False]):
        si = bulk('Si')
        name = 'si_{0:d}'.format(i)
        si.center()
        calc = GPAW(mode=PW(120), kpts=(1, 1, 2),
                    experimental={'niter_fixdensity': 2},
                    eigensolver='rmm-diis',
                    parallel={'domain': 1},
                    symmetry='off', txt=name + '.txt')
        si.calc = calc
        si.get_potential_energy()
        calc.diagonalize_full_hamiltonian(expert=expert, nbands=48)
        string = name + '.gpw'
        calc.write(string, 'all')
        wfs_e.append(calc.wfs)

    ref_revision = 133324
    # Test against values from reference revision
    epsn_n = np.array([-0.21071688, 0.05652139,
                       0.17630221, 0.17630234,
                       0.26460894])
    wfsold_G = np.array([7.51949548, 54.64611462,
                         -3.81324175, -0.14783339,
                         -0.64661614])

    kpt_u = wfs_e[0].kpt_u
    for kpt in kpt_u:
        if kpt.k == 0:
            psit = kpt.psit_nG[1, 0:5].copy()
            if wfsold_G[0] * psit[0] < 0:
                psit *= -1.
            if world.rank == 0:
                print('eps', repr(kpt.eps_n[0:5]))
                print('psit', repr(psit))
                assert np.allclose(epsn_n, kpt.eps_n[0:5], 1e-5), \
                    'Eigenvalues have changed since rev. #%d' % ref_revision
                assert np.allclose(wfsold_G, psit, 1e-5), \
                    'Wavefunctions have changed rev. #%d' % ref_revision

    # Check that expert={True, False} give
    # same result
    while len(wfs_e) > 1:
        wfs = wfs_e.pop()
        for wfstmp in wfs_e:
            for kpt, kpttmp in zip(wfs.kpt_u, wfstmp.kpt_u):
                for m, (psi_G, eps) in enumerate(zip(kpt.psit_nG, kpt.eps_n)):
                    # Have to do like this if bands are degenerate
                    booleanarray = np.abs(kpttmp.eps_n - eps) < 1e-10
                    inds = np.argwhere(booleanarray)
                    count = len(inds)
                    assert count > 0, 'Difference between eigenvalues!'

                    psitmp_nG = kpttmp.psit_nG[inds][:, 0, :]
                    fidelity = 0
                    for psitmp_G in psitmp_nG:
                        fidelity += (
                            np.abs(np.dot(psitmp_G.conj(), psi_G))**2 /
                            np.dot(psitmp_G.conj(), psitmp_G) /
                            np.dot(psi_G.conj(), psi_G))

                    assert fidelity == pytest.approx(1, abs=1e-10)
