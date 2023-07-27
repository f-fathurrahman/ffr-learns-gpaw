from sympy import *
from sympy.physics.hydrogen import R_nl

init_printing()

r = symbols("r")
Z = symbols("Z")
pprint(R_nl(1, 0, r, Z))