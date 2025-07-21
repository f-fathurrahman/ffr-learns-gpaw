import numpy as np
import scipy

# For checking ghost states
def check_diagonalize(r, h, N, beta, e_ln, n_ln, q_ln, emax, vt_in, lmax, dH_lnn, dO_lnn, ghost):
    ng = 350
    print()
    print("Diagonalizing with gridspacing h=%.3f" % h)
    R = h * np.arange(1, ng + 1)
    G = (N * R / (beta + R) + 0.5).astype(int)
    G = np.clip(G, 1, N - 2)
    R1 = np.take(r, G - 1)
    R2 = np.take(r, G)
    R3 = np.take(r, G + 1)
    x1 = (R - R2) * (R - R3) / (R1 - R2) / (R1 - R3)
    x2 = (R - R1) * (R - R3) / (R2 - R1) / (R2 - R3)
    x3 = (R - R1) * (R - R2) / (R3 - R1) / (R3 - R2)
    #
    def interpolate(f):
        f1 = np.take(f, G - 1)
        f2 = np.take(f, G)
        f3 = np.take(f, G + 1)
        return f1 * x1 + f2 * x2 + f3 * x3
    vt = interpolate(vt_in)
    print()
    print("state   all-electron     PAW")
    print("-------------------------------")
    for l in range(4):
        if l <= lmax:
            q_n = np.array([interpolate(q) for q in q_ln[l]])
            H = np.dot(np.transpose(q_n),
                       np.dot(dH_lnn[l], q_n)) * h
            S = np.dot(np.transpose(q_n),
                       np.dot(dO_lnn[l], q_n)) * h
        else:
            H = np.zeros((ng, ng))
            S = np.zeros((ng, ng))
        H.ravel()[::ng + 1] += vt + 1.0 / h**2 + l * (l + 1) / 2.0 / R**2
        H.ravel()[1::ng + 1] -= 0.5 / h**2
        H.ravel()[ng::ng + 1] -= 0.5 / h**2
        S.ravel()[::ng + 1] += 1.0
        #
        e_n, _ = scipy.linalg.eigh(H, S)
        #
        ePAW = e_n[0]
        if l <= lmax and n_ln[l][0] > 0:
            eAE = e_ln[l][0]
            print("%d%s:   %12.6f %12.6f" % (n_ln[l][0], "spdf"[l], eAE, ePAW), end="")
            if abs(eAE - ePAW) > 0.014:
                print("  GHOST-STATE!")
                ghost = True
            else:
                print()
        else:
            print("*%s:                %12.6f" % ("spdf"[l], ePAW), end="")
            if ePAW < emax:
                print("  GHOST-STATE!")
                ghost = True
            else:
                print()
    print("-------------------------------")



def divide_by_r(r, x_g, l):
    p = x_g.copy()
    p[1:] /= r[1:]
    # XXXXX go to higher order!!!!!
    if l == 0:  # l_j[self.jcorehole] == 0:
        p[0] = (p[2] + (p[1] - p[2]) * (r[0] - r[2]) / (r[1] - r[2]))
    return p

def divide_all_by_r(r, x_jg, vl_j):
    return [divide_by_r(r, x_g, l) for x_g, l in zip(x_jg, vl_j)]
