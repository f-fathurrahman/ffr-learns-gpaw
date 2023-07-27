# r is in bohr, a0=1
# Reference: https://quantummechanics.ucsd.edu/ph130a/130_notes/node233.html


# Normalization according to
#
# \int_{0}^{\infty} r^2 R^{*}_{nl}(r) R_{nl}(r) \, \mathrm{d}r = 1
#

from numpy import sqrt, exp

a0 = 1.0

def R_10(r, Z=1.0):
    return 2*(Z/a0)**(3/2) * exp(-Z*r/a0)

def R_21(r, Z=1.0):
    pref = 1/sqrt(3) * (Z/(2*a0))**(3/2)
    return pref * (Z*r/a0) * exp( -Z*r/(2*a0) )

def R_20(r, Z=1.0):
    pref = 2*(Z/(2*a0))**(3/2)
    return pref * (1 - Z*r/(2*a0)) * exp(-Z*r/(2*a0))


# XXX: Wrong normalization
def R_32(r, Z=1.0):
    pref = 2*sqrt(2)/(27*sqrt(5)) * (Z/(3*a0))**(3/2)
    return pref*(Z*r/a0)**2 * exp(-Z*r/(3*a0))

def R_32_sympy(r, Z=1.0):
    return 2*sqrt(30)*Z**2*r**2*sqrt(Z**3)*exp(-Z*r/3)/1215

# XXX: wrong normalization?
def R_31(r, Z=1.0):
    #pref = 4*sqrt(2)/3 * (Z/(3*a0))**(3/2)
    pref = 4*sqrt(2)/3 * (Z/(3*a0))**(3/2) / 3 # FIX 1/3 factor
    return pref * (Z*r/a0) * (1 - Z*r/(6*a0)) * exp(-Z*r/(3*a0))

def R_31_sympy(r, Z=1.0):
    return sqrt(6)*Z*r*(-2*Z*r/3 + 4)*sqrt(Z**3)*exp(-Z*r/3)/81


def R_30(r, Z=1.0):
    pref = 2*(Z/(3*a0))**(3/2)
    return pref * ( 1 - 2*Z*r/(3*a0) + 2*(Z*r)**2/(27*a0**2) ) * exp(-Z*r/(3*a0))



