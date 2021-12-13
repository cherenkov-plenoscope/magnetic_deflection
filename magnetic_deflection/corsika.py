import corsika_primary as cpw
import tempfile
import pandas
import numpy as np
import os

from . import light_field_characterization as lfc

MAX_ZENITH_DEG = cpw.MAX_ZENITH_DEG


def make_steering(
    run_id,
    site,
    particle_id,
    particle_energy,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
    prng,
):
    assert run_id > 0
    i8 = np.int64
    f8 = np.float64

    steering = {}
    steering["run"] = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle_energy * 0.99),
            "stop_GeV": f8(particle_energy * 1.01),
        },
        "random_seed": cpw.random.seed.make_simple_seed(seed=run_id),
    }
    steering["primaries"] = []
    for airshower_id in np.arange(1, num_showers + 1):
        az, zd = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=np.deg2rad(particle_cone_azimuth_deg),
            zenith_rad=np.deg2rad(particle_cone_zenith_deg),
            min_scatter_opening_angle_rad=np.deg2rad(0.0),
            max_scatter_opening_angle_rad=np.deg2rad(
                particle_cone_opening_angle_deg
            ),
            max_iterations=1000,
        )
        prm = {
            "particle_id": f8(particle_id),
            "energy_GeV": f8(particle_energy),
            "zenith_rad": f8(zd),
            "azimuth_rad": f8(az),
            "depth_g_per_cm2": f8(0.0),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering


def estimate_cherenkov_pool(
    corsika_primary_path,
    corsika_steering_dict,
    min_num_cherenkov_photons,
    statistics_optional={},
):
    sopt = statistics_optional
    pools = []

    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        corsika_run = cpw.CorsikaPrimary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_steering_dict,
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        )

        event_seeds = {}
        for event in corsika_run:
            evth, bunches = event
            event_id = int(evth[cpw.I.EVTH.EVENT_NUMBER])
            event_seeds[event_id] = cpw.random.seed.parse_seed_from_evth(
                evth=evth
            )
            light_field = init_light_field_from_corsika(bunches=bunches)
            num_bunches = light_field["x"].shape[0]

            if num_bunches >= min_num_cherenkov_photons:
                pool = {}
                pool["run"] = int(evth[cpw.I.EVTH.RUN_NUMBER])
                pool["event"] = event_id
                pool["particle_azimuth_deg"] = np.rad2deg(
                    evth[cpw.I.EVTH.AZIMUTH_RAD]
                )
                pool["particle_zenith_deg"] = np.rad2deg(
                    evth[cpw.I.EVTH.ZENITH_RAD]
                )
                pool["particle_energy_GeV"] = evth[cpw.I.EVTH.TOTAL_ENERGY_GEV]
                pool["cherenkov_num_photons"] = np.sum(light_field["size"])
                pool["cherenkov_num_bunches"] = num_bunches

                light_field = lfc.add_median_x_y_to_light_field(light_field)
                light_field = lfc.add_median_cx_cy_to_light_field(light_field)
                light_field = lfc.add_r_square_to_light_field_wrt_median(light_field)
                light_field = lfc.add_cos_theta_to_light_field_wrt_median(light_field)

                c = lfc.parameterize_light_field(light_field=light_field)

                if "histogram_r" in sopt:
                    c_r = lfc.histogram_r_in_light_field(
                        light_field=light_field,
                        r_bin_edges=sopt["histogram_r"]["r_bin_edges"],
                    )
                    c.update(c_r)

                if "histogram_theta" in sopt:
                    c_t = lfc.histogram_theta_in_light_field(
                        light_field=light_field,
                        theta_bin_edges=sopt["histogram_theta"]["theta_bin_edges"],
                    )
                    c.update(c_t)

                pool.update(c)
                pools.append(pool)

        return pools, event_seeds


def make_cherenkov_pools_statistics(
    site,
    particle_id,
    particle_energy,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
    min_num_cherenkov_photons,
    corsika_primary_path,
    run_id,
    prng,
    statistics_optional={},
):
    steering_dict = make_steering(
        run_id=run_id,
        site=site,
        particle_id=particle_id,
        particle_energy=particle_energy,
        particle_cone_azimuth_deg=particle_cone_azimuth_deg,
        particle_cone_zenith_deg=particle_cone_zenith_deg,
        particle_cone_opening_angle_deg=particle_cone_opening_angle_deg,
        num_showers=num_showers,
        prng=prng,
    )
    pools, event_seeds = estimate_cherenkov_pool(
        corsika_steering_dict=steering_dict,
        corsika_primary_path=corsika_primary_path,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
        statistics_optional=statistics_optional,
    )
    steering_dict["event_seeds"] = event_seeds
    return pools, steering_dict


def init_light_field_from_corsika(bunches):
    lf = {}
    lf["x"] = bunches[:, cpw.I.BUNCH.X] * cpw.CM2M  # cm to m
    lf["y"] = bunches[:, cpw.I.BUNCH.Y] * cpw.CM2M  # cm to m
    lf["cx"] = bunches[:, cpw.I.BUNCH.CX]
    lf["cy"] = bunches[:, cpw.I.BUNCH.CY]
    lf["t"] = bunches[:, cpw.I.BUNCH.TIME] * 1e-9  # ns to s
    lf["size"] = bunches[:, cpw.I.BUNCH.BSIZE]
    lf["wavelength"] = bunches[:, cpw.I.BUNCH.WVL] * 1e-9  # nm to m
    return lf
