import magnetic_deflection as mdfl
import numpy as np
import pytest
import warnings


def test_acos():
    md_arccos = mdfl.cherenkov_pool.analysis.acos_accepting_numeric_tolerance

    for v in np.linspace(-1, 1, 1000):
        assert md_arccos(v) == np.arccos(v)

    with pytest.warns(RuntimeWarning):
        v = md_arccos(-1 - 1.1e-6, eps=1e-6)

    with pytest.warns(RuntimeWarning):
        v = md_arccos(1 + 1.1e-6, eps=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(1 + 0.9e-6, eps=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(-1 - 0.9e-6, eps=1e-6)
