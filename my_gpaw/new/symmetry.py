import warnings

import numpy as np
from my_gpaw.mpi import MPIComm
from my_gpaw.new import zip
from my_gpaw.new.brillouin import IBZ, BZPoints
from my_gpaw.rotation import rotation
from my_gpaw.symmetry import Symmetry as OldSymmetry


def create_symmetries_object(atoms, ids=None, magmoms=None, parameters=None):
    ids = ids or [()] * len(atoms)
    if magmoms is None:
        pass
    elif magmoms.ndim == 1:
        ids = [id + (m,) for id, m in zip(ids, magmoms)]
    else:
        ids = [id + tuple(m) for id, m in zip(ids, magmoms)]
    symmetry = OldSymmetry(ids,
                           atoms.cell.complete(),
                           atoms.pbc,
                           **(parameters or {}))
    symmetry.analyze(atoms.get_scaled_positions())
    return Symmetries(symmetry)


def mat(rot_cc) -> str:
    """Convert 3x3 matrix to str.

    >>> mat([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    '[[-1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]'

    """
    return '[[' + '], ['.join(', '.join(f'{r:2}'
                                        for r in rot_c)
                              for rot_c in rot_cc) + ']]'


class Symmetries:
    """Wrapper for old symmetry object.  Exposes what we need now."""
    def __init__(self, symmetry):
        self.symmetry = symmetry
        self.rotation_scc = symmetry.op_scc
        self.translation_sc = symmetry.ft_sc
        self.a_sa = symmetry.a_sa
        cell_cv = symmetry.cell_cv
        self.rotation_svv = np.einsum('vc, scd, dw -> svw',
                                      np.linalg.inv(cell_cv),
                                      self.rotation_scc,
                                      cell_cv)
        self.rotation_lsmm = [
            np.array([rotation(l, r_vv) for r_vv in self.rotation_svv])
            for l in range(4)]
        self._rotations = {}

    def __len__(self):
        return len(self.rotation_scc)

    def __str__(self):
        lines = ['symmetry:',
                 f'  number of symmetries: {len(self)}']
        if self.symmetry.symmorphic:
            lines.append('  rotations: [')
            for rot_cc in self.rotation_scc:
                lines.append(f'    {mat(rot_cc)},')
        else:
            nt = self.translation_sc.any(1).sum()
            lines.append(f'  number of symmetries with translation: {nt}')
            lines.append('  rotations and translations: [')
            for rot_cc, t_c in zip(self.rotation_scc, self.translation_sc):
                a, b, c = t_c
                lines.append(f'    [{mat(rot_cc)}, '
                             f'[{a:6.3f}, {b:6.3f}, {c:6.3f}]],')
        lines[-1] = lines[-1][:-1] + ']\n'
        return '\n'.join(lines)

    def reduce(self,
               bz: BZPoints,
               comm: MPIComm = None,
               strict: bool = True) -> IBZ:
        """Find irreducible set of k-points."""
        if not (self.symmetry.time_reversal or
                self.symmetry.point_group):
            N = len(bz)
            return IBZ(self,
                       bz,
                       ibz2bz=np.arange(N),
                       bz2ibz=np.arange(N),
                       weights=np.ones(N) / N)

        (_, weight_k, sym_k, time_reversal_k, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = self.symmetry.reduce(bz.kpt_Kc, comm)

        if -1 in bz2bz_Ks:
            msg = 'Note: your k-points are not as symmetric as your crystal!'
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

        return IBZ(self, bz, ibz2bz_k, bz2ibz_K, weight_k)

    def check_positions(self, fracpos_ac):
        self.symmetry.check(fracpos_ac)

    def symmetrize_forces(self, F_av):
        return self.symmetry.symmetrize_forces(F_av)

    def rotations(self, l_j, xp=np):
        ells = tuple(l_j)
        rotation_sii = self._rotations.get(ells)
        if rotation_sii is None:
            ni = sum(2 * l + 1 for l in l_j)
            rotation_sii = np.zeros((len(self), ni, ni))
            i1 = 0
            for l in l_j:
                i2 = i1 + 2 * l + 1
                rotation_sii[:, i1:i2, i1:i2] = self.rotation_lsmm[l]
                i1 = i2
            rotation_sii = xp.asarray(rotation_sii)
            self._rotations[ells] = rotation_sii
        return rotation_sii
