from magnetic_deflection.utils import estimate_num_bins_to_contain_quantile
import numpy as np


def test_basics():
    prng = np.random.Generator(np.random.PCG64(12))
    counts = prng.integers(low=0, high=1000, size=1000)

    b = estimate_num_bins_to_contain_quantile(counts=counts, q=0)
    assert b == 0.0

    b = estimate_num_bins_to_contain_quantile(counts=counts, q=1)
    assert b == len(counts)

    ones = np.ones(1000)
    for q in np.linspace(0, 1, 14):
        b = estimate_num_bins_to_contain_quantile(counts=ones, q=q)
        assert b == len(ones) * q

    zeros = np.zeros(100)
    b = estimate_num_bins_to_contain_quantile(counts=zeros, q=q)
    assert np.isnan(b)

    linear = np.arange(1000)
    q = 0.5
    b = estimate_num_bins_to_contain_quantile(counts=linear, q=q)
    assert b < q * len(linear)

    b = estimate_num_bins_to_contain_quantile(
        counts=linear, q=q, mode="max_num_bins"
    )
    assert b > q * len(linear)


def test_aperture():
    prng = np.random.Generator(np.random.PCG64(12))

    ones = np.ones(1000)

    for bin_aperture_width in [0.1, 1.0, 2.0]:
        for q in np.linspace(0, 1, 14):
            b, a = estimate_num_bins_to_contain_quantile(
                counts=ones,
                q=q,
                bin_apertures=bin_aperture_width * ones,
            )
            assert b == len(ones) * q
            np.testing.assert_almost_equal(
                a, bin_aperture_width * len(ones) * q
            )
