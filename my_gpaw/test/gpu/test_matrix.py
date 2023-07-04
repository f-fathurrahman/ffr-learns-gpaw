import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, as_xp


@pytest.mark.gpu
@pytest.mark.serial
def test_zyrk():
    a = np.array([[1, 1 + 2j, 2], [1, 0.5j, -1 - 0.5j]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b.tril2full()
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b2 = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b2.tril2full()
    c = b2.to_cpu()
    assert (c.data == b.data).all()


@pytest.mark.gpu
@pytest.mark.serial
def test_eigh():
    H1 = Matrix(2, 2, data=np.array([[2, 42.1 + 42.1j], [0.1 - 0.1j, 3]]))
    S1 = Matrix(2, 2, data=np.array([[1, 42.1 + 42.2j], [0.1 - 0.2j, 0.9]]))
    H2 = Matrix(2, 2, data=cp.asarray(H1.data))
    S2 = Matrix(2, 2, data=cp.asarray(S1.data))

    E1 = H1.eigh(S1)

    S0 = S1.copy()
    S0.tril2full()

    E2 = H2.eigh(S2)
    assert as_xp(E2, np) == pytest.approx(E1)

    C1 = H1.data
    C2 = H2.to_cpu().data

    # Check that eigenvectors are parallel:
    X = C1.conj() @ S0.data @ C2.T
    assert abs(X) == pytest.approx(np.eye(2))
