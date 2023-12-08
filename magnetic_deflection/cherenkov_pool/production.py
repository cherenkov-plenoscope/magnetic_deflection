import corsika_primary as cpw
import os
import numpy as np
import tempfile
import atmospheric_cherenkov_response as acr
from . import analysis
from .. import spherical_coordinates


def make_example_steering():
    return make_steering(
        run_id=1337,
        site=acr.sites.init("lapalma"),
        particle_id=3.0,
        particle_energy_start_GeV=1.0,
        particle_energy_stop_GeV=5.0,
        particle_energy_power_slope=-2,
        particle_cone_azimuth_deg=0.0,
        particle_cone_zenith_deg=0.0,
        particle_cone_opening_angle_deg=70.0,
        num_showers=1000,
    )


def make_steering(
    run_id,
    site,
    particle_id,
    particle_energy_start_GeV,
    particle_energy_stop_GeV,
    particle_energy_power_slope,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
):
    assert run_id > 0
    i8 = np.int64
    f8 = np.float64

    prng = np.random.Generator(np.random.PCG64(seed=run_id))

    steering = {}
    steering["run"] = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle_energy_start_GeV * 0.99),
            "stop_GeV": f8(particle_energy_stop_GeV * 1.01),
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
        energy_GeV = cpw.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=particle_energy_start_GeV,
            upper_limit=particle_energy_stop_GeV,
            power_slope=particle_energy_power_slope,
            num_samples=1,
        )[0]
        prm = {
            "particle_id": f8(particle_id),
            "energy_GeV": f8(energy_GeV),
            "zenith_rad": f8(zd),
            "azimuth_rad": f8(az),
            "depth_g_per_cm2": f8(0.0),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering


def estimate_cherenkov_pool(corsika_primary_path, corsika_steering_dict):
    pools = []
    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        with cpw.CorsikaPrimary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_steering_dict,
            particle_output_path=os.path.join(tmp_dir, "corsika.par.dat"),
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        ) as corsika_run:
            for event in corsika_run:
                evth, bunch_reader = event
                corsika_bunches = np.vstack([b for b in bunch_reader])

                light_field = init_light_field_from_corsika_bunches(
                    corsika_bunches=corsika_bunches
                )

                pool = {}
                pool["run"] = int(evth[cpw.I.EVTH.RUN_NUMBER])
                pool["event"] = int(evth[cpw.I.EVTH.EVENT_NUMBER])

                par_uxyz = particle_direction_uxyz(evth=evth)
                pool["particle_cx_rad"] = par_uxyz[0]
                pool["particle_cy_rad"] = par_uxyz[1]
                pool["particle_energy_GeV"] = evth[cpw.I.EVTH.TOTAL_ENERGY_GEV]
                pool["cherenkov_num_photons"] = np.sum(light_field["size"])
                pool["cherenkov_num_bunches"] = light_field["x"].shape[0]
                pool[
                    "cherenkov_maximum_asl_m"
                ] = estimate_cherenkov_maximum_asl_m(
                    corsika_bunches=corsika_bunches
                )
                pool.update(analysis.init(light_field=light_field))
                pools.append(pool)

        return pools


def estimate_cherenkov_maximum_asl_m(corsika_bunches):
    if len(corsika_bunches) == 0:
        return float("nan")
    else:
        return cpw.CM2M * np.median(
            corsika_bunches[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
        )


def init_light_field_from_corsika_bunches(corsika_bunches):
    cb = corsika_bunches
    lf = {}
    lf["x"] = cb[:, cpw.I.BUNCH.X_CM] * cpw.CM2M  # cm to m
    lf["y"] = cb[:, cpw.I.BUNCH.Y_CM] * cpw.CM2M  # cm to m
    lf["cx"] = cb[:, cpw.I.BUNCH.CX_RAD]
    lf["cy"] = cb[:, cpw.I.BUNCH.CY_RAD]
    lf["t"] = cb[:, cpw.I.BUNCH.TIME_NS] * 1e-9  # ns to s
    lf["size"] = cb[:, cpw.I.BUNCH.BUNCH_SIZE_1]
    lf["wavelength"] = cb[:, cpw.I.BUNCH.WAVELENGTH_NM] * 1e-9  # nm to m
    return lf


def particle_direction_uxyz(evth):
    momentum = np.zeros(3, dtype=np.float64)
    momentum[0] = evth[cpw.I.EVTH.PX_MOMENTUM_GEV_PER_C]
    momentum[1] = evth[cpw.I.EVTH.PY_MOMENTUM_GEV_PER_C]
    momentum[2] = evth[cpw.I.EVTH.PZ_MOMENTUM_GEV_PER_C]
    particle_direction_from_momentum = momentum / np.linalg.norm(momentum)

    particle_azimuth_deg = np.rad2deg(evth[cpw.I.EVTH.AZIMUTH_RAD])
    particle_zenith_deg = np.rad2deg(evth[cpw.I.EVTH.ZENITH_RAD])

    particle_direction_from_alt_az = np.array(
        spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=particle_azimuth_deg, zenith_deg=particle_zenith_deg
        )
    )

    particle_direction_delta_vector = (
        particle_direction_from_momentum - particle_direction_from_alt_az
    )
    particle_direction_delta_rad = np.linalg.norm(
        particle_direction_delta_vector
    )
    particle_direction_delta_deg = np.rad2deg(particle_direction_delta_rad)
    assert particle_direction_delta_deg < 1e-2

    return particle_direction_from_momentum
