# creates: acf_example.png
import numpy as np
import matplotlib.pyplot as plt
from my_gpaw25.core import UGDesc

alpha = 4.0
rcut = 2.0
l = 0
gauss = (l, rcut, lambda r: (4 * np.pi)**0.5 * np.exp(-alpha * r**2))
grid = UGDesc(cell=[4.0, 2.0, 2.0], size=[40, 20, 20])

pos = [
    [0.25, 0.5, 0.5],
    [0.75, 0.5, 0.5]
]

acf_aR = grid.atom_centered_functions([[gauss], [gauss]], pos)

coef_as = acf_aR.empty(dims=(2,))

coef_as[0] = [[1], [-1]]
coef_as[1] = [[2], [1]]
print(coef_as.data, coef_as[0])

f_sR = grid.zeros(2)
acf_aR.add_to(f_sR, coef_as)
x = grid.xyz()[:, 10, 10, 0]
y1, y2 = f_sR.data[:, :, 10, 10]

ax = plt.subplot(1, 1, 1)
ax.plot(x, y1, 'o-')
ax.plot(x, y2, 'x-')
plt.show()