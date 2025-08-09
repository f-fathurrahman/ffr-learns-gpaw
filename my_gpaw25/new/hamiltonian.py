from __future__ import annotations

from my_gpaw25.core import UGArray
from my_gpaw25.core.arrays import DistributedArrays as XArray


class Hamiltonian:
    def apply(self,
              vt_sR: UGArray,
              dedtaut_sR: UGArray | None,
              ibzwfs,
              D_asii,
              psit_nG: XArray,
              out: XArray,
              spin: int) -> XArray:
        typ_str = str(type(self)).replace("<", "").replace(">", "")
        print("--- New.Hamiltonian.apply is called, type of self = ", typ_str)
        # many calls
        #breakpoint() #???
        #raise SystemError("Hello ...") # useful to get call trace
        # where is non-local potential updated?
        self.apply_local_potential(vt_sR[spin], psit_nG, out)
        #
        if dedtaut_sR is not None:
            self.apply_mgga(dedtaut_sR[spin], psit_nG, out)
        self.apply_orbital_dependent(ibzwfs, D_asii, psit_nG, spin, out)
        return out

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: XArray,
                              out: XArray) -> None:
        raise NotImplementedError

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: XArray,
                   vt_nG: XArray) -> None:
        raise NotImplementedError

    def apply_orbital_dependent(self,
                                ibzwfs,
                                D_asii,
                                psit_nG: XArray,
                                spin: int,
                                out: XArray) -> None:
        typ_str = str(type(self)).replace("<", "").replace(">", "")
        print("in apply_orbital_dependent: should only pass, type(self) = ", typ_str)
        pass

    def create_preconditioner(self, blocksize):
        raise NotImplementedError
