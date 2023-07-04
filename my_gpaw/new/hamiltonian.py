class Hamiltonian:
    def apply(self, vt_sR, psit_nR, out, spin):
        raise NotImplementedError

    def create_preconditioner(self, blocksize):
        raise NotImplementedError
