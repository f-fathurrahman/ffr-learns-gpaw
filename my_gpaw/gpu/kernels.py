from my_gpaw.gpu import cupy_is_fake
import _gpaw
try:
    from _gpaw_gpu import pwlfc_expand_gpu as pwacf_expand
except ImportError:
    def pwacf_expand(f_Gs, emiGR_Ga, Y_GL,
                     l_s, a_J, s_J,
                     cc, f_GI, I_J):
        assert cupy_is_fake
        _gpaw.pwlfc_expand(f_Gs._data, emiGR_Ga._data, Y_GL._data,
                           l_s._data, a_J._data, s_J._data,
                           cc, f_GI._data)
