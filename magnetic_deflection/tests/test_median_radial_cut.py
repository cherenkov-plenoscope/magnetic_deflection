import numpy as np
import magnetic_deflection as mdfl


def test_median_radial():
    prng = np.random.Generator(np.random.PCG64(0))
    x = prng.normal(loc=2.0, scale=1.0, size=1000)
    y = prng.normal(loc=3.0, scale=1.5, size=1000)

    mask = mdfl.light_field_characterization.percentile_indices_wrt_median_radial(
        value_dim0=x, value_dim1=y, percentile=50.0,
    )

    num_valid = np.sum(mask)
    assert num_valid >= 499
    assert num_valid <= 501


def test_median_radial_in_uniform():
    N = 1000 * 10
    percentile = 50
    fraction = percentile / 100.0
    prng = np.random.Generator(np.random.PCG64(0))
    x = prng.uniform(low=-1.0, high=1.0, size=N)
    y = prng.uniform(low=-1.0, high=1.0, size=N)

    mask = mdfl.light_field_characterization.percentile_indices_wrt_median_radial(
        value_dim0=x, value_dim1=y, percentile=percentile,
    )

    num_valid = np.sum(mask)
    assert num_valid >= N * fraction - 1
    assert num_valid <= N * fraction + 1

    r = np.hypot(x, y)
    r_max = np.max(r[mask])

    Area_outer_square = 4
    Area_inner_circle = Area_outer_square * fraction
    r_expected = np.sqrt(Area_inner_circle / np.pi)

    assert np.abs(r_max - r_expected) < 0.01
