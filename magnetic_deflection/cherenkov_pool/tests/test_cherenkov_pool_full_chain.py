import magnetic_deflection as mdfl
import numpy as np


def test_full_analysis():
    prng = np.random.Generator(np.random.PCG64(9))

    num_showers = 50
    for cc in range(num_showers):
        truth = {}
        truth["x"] = prng.uniform(low=-1e3, high=1e3)
        truth["y"] = prng.uniform(low=-1e3, high=1e3)
        truth["radius"] = prng.uniform(low=0.5e2, high=1.5e2)
        truth["direction"] = mdfl.cherenkov_pool.testing.draw_direction(
            prng=prng,
            max_zenith_distance=np.deg2rad(70),
        )
        truth["half_angle"] = np.deg2rad(prng.uniform(low=5.0, high=30.0))
        truth["t"] = prng.uniform(low=-1e-6, high=1e-6)
        truth["t_std"] = prng.uniform(low=1e-9, high=10e-9)

        size = int(prng.uniform(low=5e3, high=5e4))

        lf, _truth = mdfl.cherenkov_pool.testing.draw_dummy_light_field(
            prng=prng,
            size=size,
            radius=truth["radius"],
            x=truth["x"],
            y=truth["y"],
            half_angle=truth["half_angle"],
            cx=truth["direction"][0],
            cy=truth["direction"][1],
            t=truth["t"],
            t_std=truth["t_std"],
        )

        truth.update(_truth)

        result = mdfl.cherenkov_pool.analysis.init(light_field=lf)

        assert np.abs(result["cherenkov_x_m"] - truth["x"]) < 10.0
        assert np.abs(result["cherenkov_y_m"] - truth["y"]) < 10.0
        assert (
            np.abs(result["cherenkov_radius90_m"] - truth["radius"])
            < 0.15 * truth["radius"]
        )

        assert np.abs(
            result["cherenkov_cx_rad"] - truth["direction"][0]
        ) < np.deg2rad(2.0)
        assert np.abs(
            result["cherenkov_cy_rad"] - truth["direction"][1]
        ) < np.deg2rad(2.0)
        assert (
            np.abs(result["cherenkov_half_angle90_rad"] - truth["half_angle"])
            < 0.15 * truth["half_angle"]
        )

        assert np.abs(result["cherenkov_t_s"] - truth["t"]) < 10e-9
        assert (
            np.abs(result["cherenkov_duration50_s"] - truth["t_std"])
            < 0.5 * truth["t_std"]
        )
