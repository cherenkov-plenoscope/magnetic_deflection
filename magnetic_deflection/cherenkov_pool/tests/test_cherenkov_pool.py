import magnetic_deflection as mdfl
import numpy as np
import pytest
import warnings


def test_acos():
    md_arccos = mdfl.cherenkov_pool.analysis.acos_accepting_numeric_tolerance

    vvv = np.linspace(-1, 1, 1000)
    for v in vvv:
        assert md_arccos(v) == np.arccos(v)
    np.testing.assert_array_equal(md_arccos(v), np.arccos(v))

    lll = -1 - 1.1e-6
    with pytest.warns(RuntimeWarning):
        v = md_arccos(lll, eps=1e-6)
    with pytest.warns(RuntimeWarning):
        v = md_arccos(np.array([lll, 0.0]), eps=1e-6)

    ppp = 1 + 1.1e-6
    with pytest.warns(RuntimeWarning):
        v = md_arccos(ppp, eps=1e-6)
    with pytest.warns(RuntimeWarning):
        v = md_arccos(np.array([ppp, 0.0]), eps=1e-6)

    cccppp = 1 + 0.9e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(cccppp, eps=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(np.array([0.0, cccppp]), eps=1e-6)

    ccclll = -1 - 0.9e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(ccclll, eps=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = md_arccos(np.array([0.0, 0.1, ccclll]), eps=1e-6)


def test_radius_wrt_center_minimal():
    rr = mdfl.cherenkov_pool.analysis.make_radius_wrt_center_position(
        photon_x_m=np.ones(5),
        photon_y_m=np.zeros(5),
        center_x_m=0.0,
        center_y_m=0.0,
    )
    assert len(rr) == 5
    np.testing.assert_array_equal(rr, np.ones(5))


def test_radius_wrt_center():
    prng = np.random.Generator(np.random.PCG64(9))

    num_showers = 50
    for cc in range(num_showers):
        core_x = prng.uniform(low=-10, high=10, size=1)
        core_y = prng.uniform(low=-10, high=10, size=1)
        num_photons = int(prng.uniform(low=5e3, high=5e4, size=1))

        lf, truth = mdfl.cherenkov_pool.testing.draw_xy_in_disc(
            prng=prng,
            x=core_x,
            y=core_y,
            size=num_photons,
            radius=1e2,
        )

        rec_core_x = np.median(lf["x"])
        rec_core_y = np.median(lf["y"])

        assert np.abs(rec_core_x - core_x) < 1.0
        assert np.abs(rec_core_y - core_y) < 1.0

        rec_radii = (
            mdfl.cherenkov_pool.analysis.make_radius_wrt_center_position(
                photon_x_m=lf["x"],
                photon_y_m=lf["y"],
                center_x_m=rec_core_x,
                center_y_m=rec_core_y,
            )
        )
        for i in range(num_photons):
            assert np.abs(rec_radii[i] - truth["radii"][i]) < 1.0


def test_make_cos_theta_wrt_center_direction():
    prng = np.random.Generator(np.random.PCG64(9))

    num_showers = 50
    for cc in range(num_showers):
        prm_pointing = mdfl.cherenkov_pool.testing.draw_direction(
            prng=prng, max_zenith_distance=np.deg2rad(70)
        )
        num_photons = int(prng.uniform(low=5e3, high=5e4, size=1))

        lf, truth = mdfl.cherenkov_pool.testing.draw_cxcy_in_cone(
            prng=prng,
            cx=prm_pointing[0],
            cy=prm_pointing[1],
            size=num_photons,
            half_angle=np.deg2rad(15.0),
        )

        rec_cx = np.median(lf["cx"])
        rec_cy = np.median(lf["cy"])

        assert np.abs(rec_cx - prm_pointing[0]) < np.deg2rad(2.0)
        assert np.abs(rec_cy - prm_pointing[1]) < np.deg2rad(2.0)

        rec_theta = (
            mdfl.cherenkov_pool.analysis.make_theta_wrt_center_direction(
                photon_cx_rad=lf["cx"],
                photon_cy_rad=lf["cy"],
                center_cx_rad=rec_cx,
                center_cy_rad=rec_cy,
            )
        )
        for i in range(num_photons):
            assert np.abs(rec_theta[i] - truth["theta"][i]) < np.deg2rad(2.0)
