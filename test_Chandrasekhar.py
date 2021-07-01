from Chandrasekhar import G
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st


def test_known_values_Chandrasekhar_G():
    large_x = 1e30
    assert np.isclose(G(large_x), 0)

    small_x = 1e-16
    assert np.isclose(G(small_x), 0)
    assert isinstance(G(small_x), np.ndarray)

    assert np.isclose(G(0), 0)
    assert isinstance(G(0), np.ndarray)


def test_known_values_as_numpy_array_Chandrasekhar_G():
    x = np.array([1e30, 1e-16, 0])
    assert np.allclose(G(x), 0)
    assert isinstance(G(x), np.ndarray)
    assert G(x).size == 3


def test_regression_G():
    x = np.linspace(-5, 5, 15)
    y = G(x)
    stored_values = np.array(
        [
            -0.02,
            -0.02722222,
            -0.03919953,
            -0.06119047,
            -0.10595478,
            -0.18306813,
            -0.19961233,
            0.0,
            0.19961233,
            0.18306813,
            0.10595478,
            0.06119047,
            0.03919953,
            0.02722222,
            0.02,
        ]
    )

    assert_allclose(y, stored_values)


@given(
    x=st.floats(),
)
def test_properties_G(x):
    result = G(x)
    assert np.isfinite(result)
