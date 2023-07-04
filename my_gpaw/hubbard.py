from typing import Tuple

import numpy as np
import ase.units as units

from my_gpaw.typing import Array2D, ArrayLike2D
from my_gpaw.utilities import pack2, unpack2


def parse_hubbard_string(type: str) -> Tuple[str, 'HubbardU']:

    # Parse DFT+U parameters from type-string:
    # Examples: "type:l,U" or "type:l,U,scale"
    type, lus = type.split(':')
    if type == '':
        type = 'paw'

    l = []
    U = []
    scale = []

    for lu in lus.split(';'):  # Multiple U corrections
        l_, u_, scale_ = (lu + ',,').split(',')[:3]
        l.append('spdf'.find(l_))
        U.append(float(u_) / units.Hartree)
        if scale_:
            scale.append(bool(int(scale_)))
        else:
            scale.append(True)
    return type, HubbardU(U, l, scale)


class HubbardU:
    def __init__(self, U, l, scale=1):
        self.scale = scale
        self.U = U
        self.l = l

    def _tuple(self):
        # Tests use this method to compare to expected values
        return (self.l, self.U, self.scale)

    def calculate(self, setup, D_sp):
        e_xc = 0.0
        dH_sp = np.zeros_like(D_sp)
        for l, U, scale in zip(self.l, self.U, self.scale):
            e1_xc, dH1_sp = hubbard(
                setup.l_j, setup.lq, D_sp,
                l=l, U=U, scale=scale)

            e_xc += e1_xc
            dH_sp += dH1_sp
        return e_xc, dH_sp

    def descriptions(self):
        for U, l, scale in zip(self.U, self.l, self.scale):
            yield f'Hubbard: {{U: {U * units.Ha},  # eV\n'
            yield f'          l: {l},\n'
            yield f'          scale: {bool(scale)}}}'


def hubbard(l_j, lq,
            D_sp,
            l: int,
            U: float,
            scale: bool) -> Tuple[float, ArrayLike2D]:
    nspins = len(D_sp)

    nl = np.where(np.equal(l_j, l))[0]
    nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()

    e_xc = 0.0
    dH_sp = []

    s = 0
    for D_p in D_sp:
        N_mm, V = aoom(l_j, lq, unpack2(D_p), l, scale)
        N_mm = N_mm / 2 * nspins

        if nspins == 4:
            N_mm = N_mm / 2.0
            if s == 0:
                Eorb = U / 2. * (N_mm -
                                 0.5 * np.dot(N_mm, N_mm)).trace()

                Vorb = U / 2. * (np.eye(2 * l + 1) - N_mm)

            else:
                Eorb = U / 2. * (-0.5 * np.dot(N_mm, N_mm)).trace()

                Vorb = -U / 2. * N_mm
        else:
            Eorb = U / 2. * (N_mm -
                             np.dot(N_mm, N_mm)).trace()

            Vorb = U * (0.5 * np.eye(2 * l + 1) - N_mm)

        e_xc += Eorb
        if nspins == 1:
            # add contribution of other spin manyfold
            e_xc += Eorb

        if len(nl) == 2:
            mm = (2 * np.array(l_j) + 1)[0:nl[1]].sum()

            V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            V[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            V[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
            V[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
        else:
            V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb

        dH_sp.append(pack2(V))
        s += 1

    return e_xc, dH_sp


def aoom(l_j, lq,
         DM: Array2D,
         l: int,
         scale: bool = True) -> Tuple[Array2D, Array2D]:
    """Atomic Orbital Occupation Matrix.

    Determine the Atomic Orbital Occupation Matrix (aoom) for a
    given l-quantum number.

    This operation, takes the density matrix (DM), which for
    example is given by unpack2(D_asq[i][spin]), and corrects for
    the overlap between the selected orbitals (l) upon which the
    the density is expanded (ex <p|p*>,<p|p>,<p*|p*> ).

    Returned is only the "corrected" part of the density matrix,
    which represents the orbital occupation matrix for l=2 this is
    a 5x5 matrix.
    """
    nl = np.where(np.equal(l_j, l))[0]
    V = np.zeros(np.shape(DM))
    if len(nl) == 2:
        aa = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2
        bb = nl[1] * len(l_j) - (nl[1] - 1) * nl[1] // 2
        ab = aa + nl[1] - nl[0]

        if not scale:
            lq_a = lq[aa]
            lq_ab = lq[ab]
            lq_b = lq[bb]
        else:
            lq_a = 1
            lq_ab = lq[ab] / lq[aa]
            lq_b = lq[bb] / lq[aa]

        # and the correct entrances in the DM
        nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()
        mm = (2 * np.array(l_j) + 1)[0:nl[1]].sum()

        # finally correct and add the four submatrices of NC_DM
        A = DM[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] * lq_a
        B = DM[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] * lq_ab
        C = DM[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] * lq_ab
        D = DM[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] * lq_b

        V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] = lq_a
        V[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] = lq_ab
        V[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] = lq_ab
        V[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] = lq_b

        return A + B + C + D, V
    else:
        assert len(nl) == 1
        assert l_j[-1] == l
        nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()
        A = DM[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] * lq[-1]
        V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] = lq[-1]
        return A, V
