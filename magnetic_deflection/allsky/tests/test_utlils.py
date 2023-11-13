import magnetic_deflection as mdfl
import numpy as np


def assert_approx(a, b, eps=1e-6):
    assert abs(a - b) <= eps


def test_gauss1d():
    assert_approx(1.0, mdfl.allsky.gauss1d(x=0, mean=0.0, sigma=1.0))
    assert_approx(0.0, mdfl.allsky.gauss1d(x=100, mean=0.0, sigma=1.0))
    assert_approx(0.0, mdfl.allsky.gauss1d(x=-100, mean=0.0, sigma=1.0))
    assert_approx(0.60, mdfl.allsky.gauss1d(x=1, mean=0.0, sigma=1.0), eps=0.01)
    assert_approx(0.60, mdfl.allsky.gauss1d(x=-1, mean=0.0, sigma=1.0), eps=0.01)


def test_weighted_avg_and_std_all_weights_one():
    prng = np.random.Generator(np.random.PCG64(43))

    for i in range(10):
        values = prng.uniform(size=100)
        weights = np.ones(100)
        a, s = mdfl.allsky.weighted_avg_and_std(values=values, weights=weights)
        assert_approx(a, np.mean(values))
        assert_approx(s, np.std(values))


def test_weighted_avg_and_std():
    values = np.linspace(0, 1, 1000)
    weights = np.linspace(0, 1, 1000)
    a, s = mdfl.allsky.weighted_avg_and_std(values=values, weights=weights)
    assert_approx(a, 2/3, eps=1e-3)

    values = np.linspace(0, 1, 1000)
    weights = np.linspace(1, 0, 1000)
    a, s = mdfl.allsky.weighted_avg_and_std(values=values, weights=weights)
    assert_approx(a, 1/3, eps=1e-3)
