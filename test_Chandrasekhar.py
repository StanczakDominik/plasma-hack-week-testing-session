from Chandrasekhar import G
import numpy as np


def test_known_values_Chandrasekhar_G():
    large_x = 1e30
    assert np.isclose(G(large_x), 0)

    small_x = 1e-16
    assert np.isclose(G(small_x), 0)
