import numpy as np
from my_gpaw.xc import XC


class XCFunctional:
    def __init__(self, params: dict, ncomponents: int):
        if isinstance(params, (dict, str)):
            self.xc = XC(params, collinear=(ncomponents < 4))
        else:
            self.xc = params
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')
        self.type = self.xc.type

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate(self, nt_sr, vxct_sr) -> float:
        if nt_sr.xp is np:
            vxct_sr.data[:] = 0.0
            return self.xc.calculate(nt_sr.desc._gd, nt_sr.data, vxct_sr.data)
        vxct_np_sr = np.zeros(vxct_sr.data.shape)
        exc = self.xc.calculate(nt_sr.desc._gd, nt_sr.data.get(), vxct_np_sr)
        vxct_sr.data[:] = vxct_sr.xp.asarray(vxct_np_sr)
        return exc

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self):
        return self.name
