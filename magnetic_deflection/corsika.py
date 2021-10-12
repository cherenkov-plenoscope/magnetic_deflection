import corsika_primary_wrapper as cpw
import tempfile
import pandas
import numpy as np
import os

from . import light_field_characterization as lfc


def make_steering(
    run_id,
    site,
    primary_particle_id,
    primary_energy,
    primary_cone_azimuth_deg,
    primary_cone_zenith_deg,
    primary_cone_opening_angle_deg,
    num_events,
    prng,
):
    steering = {}
    steering["run"] = {
        "run_id": int(run_id),
        "event_id_of_first_event": 1,
        "observation_level_asl_m": site["observation_level_asl_m"],
        "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
        "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
        "atmosphere_id": site["atmosphere_id"],
    }
    steering["primaries"] = []
    for event_id in range(num_events):
        az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=np.deg2rad(primary_cone_azimuth_deg),
            zenith_rad=np.deg2rad(primary_cone_zenith_deg),
            min_scatter_opening_angle_rad=np.deg2rad(0.0),
            max_scatter_opening_angle_rad=np.deg2rad(primary_cone_opening_angle_deg),
            max_zenith_rad=np.deg2rad(90),
        )
        prm = {
            "particle_id": int(primary_particle_id),
            "energy_GeV": float(primary_energy),
            "zenith_rad": zd,
            "azimuth_rad": az,
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw.simple_seed(event_id + run_id * num_events),
        }
        steering["primaries"].append(prm)
    return steering


def estimate_cherenkov_pool(
    corsika_primary_steering, corsika_primary_path, min_num_cherenkov_photons, outlier_percentile
):
    pools = []

    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        corsika_run = cpw.CorsikaPrimary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_primary_steering,
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        )

        for idx, airshower in enumerate(corsika_run):
            corsika_event_header, photon_bunches = airshower
            all_light_field = lfc.init_light_field_from_corsika(
                bunches=photon_bunches
            )

            mask_inlier = lfc.mask_inlier_in_light_field_geometry(
                light_field=all_light_field,
                percentile=outlier_percentile,
            )

            light_field = all_light_field[mask_inlier]

            num_bunches = light_field.shape[0]
            ceh = corsika_event_header

            if num_bunches >= min_num_cherenkov_photons:
                pool = {}
                pool["primary_azimuth_rad"] = ceh[cpw.I_EVTH_AZIMUTH_RAD]
                pool["primary_zenith_rad"] = ceh[cpw.I_EVTH_ZENITH_RAD]
                pool["num_photons"] = np.sum(light_field["size"])
                pool["num_bunches"] = num_bunches

                c = lfc.parameterize_light_field(light_field=light_field)
                pool.update(c)
                pools.append(pool)

        return pools