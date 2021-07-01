from scipy.special import erf
import numpy as np


def G(x):
    erf_derivative = 2 * np.exp(-(x ** 2)) / np.sqrt(np.pi)
    return (erf(x) / x ** 2 - erf_derivative / x) / 2
