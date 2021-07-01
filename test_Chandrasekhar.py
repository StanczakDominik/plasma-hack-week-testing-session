from Chandrasekhar import G
import numpy as np


def test_known_values_Chandrasekhar_G():
    large_x = 1e30
    assert np.isclose(G(large_x), 0)

    small_x = 1e-16
    assert np.isclose(G(small_x), 0)

    assert np.isclose(G(0), 0)

def test_known_values_as_numpy_array_Chandrasekhar_G():
    x = np.array([1e30, 1e-16, 0])
    assert np.allclose(G(x), 0)
