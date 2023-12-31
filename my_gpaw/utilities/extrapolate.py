import sys
import numpy as np
from ase.parallel import paropen


def extrapolate(x, y, n=-1.5, plot=0, reg=0, txt=None):
    """Extrapolation tool. Mainly intended for RPA correlation energies,
    but could be useful for other purposes. Fits a straight line to an
    expression of the form: y=b + alpha*x**n and extrapolates the result
    to infinite x. reg=N gives linear regression using the last N points in
    x. reg should be larger than 2"""

    if txt is None:
        f = sys.stdout
    elif isinstance(txt, str):
        f = paropen(txt, 'a')
    else:
        f = txt
    assert len(x) == len(y)
    ext = []
    print('Two-point extrapolation:', file=f)
    for i in range(len(x) - 1):
        alpha = (y[i] - y[i + 1]) / (x[i]**n - x[i + 1]**n)
        ext.append(y[i + 1] - alpha * x[i + 1]**n)
        print('    ', x[i], '-', x[i + 1], ':', ext[-1], file=f)
    print(file=f)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x**n, y, 'o-', label='Data')
        plt.xticks(x**n, [int(e) for e in x])
        plt.axis([0, None, None, None])
    if reg > 2:
        a = x[-reg:]**n
        b = y[-reg:]
        N = reg
        delta = N * np.sum(a**2) - (np.sum(a))**2
        A = (np.sum(a**2) * np.sum(b) - np.sum(a) * np.sum(a * b)) / delta
        B = (N * np.sum(a * b) - np.sum(a) * np.sum(b)) / delta
        sigma_y = (1 / (N - 2) * np.sum((b - A - B * a)**2))**0.5
        sigma_A = sigma_y * (np.sum(a**2) / delta)**0.5

        print('Linear regression using last %s points:' % N, file=f)
        print('    Extrapolated result:', A, file=f)
        print('    Uncertainty:', sigma_A, file=f)
        print(file=f)
        if plot:
            print([a[0], 0], [A + B * a[0], A])
            plt.plot([a[0], 0], [A + B * a[0], A], '--', label='Regression')
            plt.legend(loc='upper left')
    else:
        A = 0
        B = 0
        sigma_A = 0
    if plot:
        plt.show()
        plt.plot(x[1:], ext, 'o-', label='Two-point extrapolation')
        if reg > 2:
            plt.plot([x[-reg], x[-1]], [A, A], '--', label='Regression')
            plt.errorbar(x[-2], A, yerr=sigma_A,
                         elinewidth=2.0, capsize=5, c='g')

        plt.legend(loc='lower right')
        plt.show()
    if txt is not None:
        f.close()
    return ext, A, B, sigma_A
