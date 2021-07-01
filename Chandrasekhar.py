from scipy.special import erf
import numpy as np


def G(x):
    x = np.asarray(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        erf_derivative = 2 * np.exp(-x**2) / np.sqrt(np.pi)
        output = (erf(x) / x **2 - erf_derivative / x) / 2
    output = np.where(x == 0, 0, output)
    return output
