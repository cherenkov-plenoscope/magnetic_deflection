import corsika_primary_wrapper as cpw
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
    seed_maker_and_checker = cpw.random_seed.CorsikaRandomSeed(
        NUM_DIGITS_RUN_ID=4,
        NUM_DIGITS_AIRSHOWER_ID=5,
    )
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
    for airshower_id in np.arange(1, num_showers + 1):
        az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
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
            "particle_id": int(particle_id),
            "energy_GeV": float(particle_energy),
            "zenith_rad": zd,
            "azimuth_rad": az,
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw.simple_seed(
                seed=seed_maker_and_checker.random_seed_based_on(
                    run_id=run_id,
                    airshower_id=airshower_id,
                ),
            ),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering


def estimate_cherenkov_pool(
    corsika_primary_steering,
    corsika_primary_path,
    min_num_cherenkov_photons,
    density_cut,
):
    pools = []

    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        corsika_run = cpw.CorsikaPrimary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_primary_steering,
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        )
        explicit_steering = extract_explicit_steering(corsika_run=corsika_run)

        for idx, shower in enumerate(corsika_run):
            corsika_event_header, photon_bunches = shower
            all_light_field = init_light_field_from_corsika(
                bunches=photon_bunches
            )
            num_all_bunches = len(all_light_field)

            mask_inlier = lfc.light_field_density_cut(
                light_field=all_light_field, density_cut=density_cut,
            )

            light_field = all_light_field[mask_inlier]
            del(all_light_field)

            num_bunches = light_field.shape[0]
            ceh = corsika_event_header

            if num_bunches >= min_num_cherenkov_photons:
                pool = {}
                pool["run"] = int(ceh[cpw.I_EVTH_RUN_NUMBER])
                pool["event"] = int(ceh[cpw.I_EVTH_EVENT_NUMBER])
                pool["particle_azimuth_deg"] = np.rad2deg(
                    ceh[cpw.I_EVTH_AZIMUTH_RAD]
                )
                pool["particle_zenith_deg"] = np.rad2deg(
                    ceh[cpw.I_EVTH_ZENITH_RAD]
                )
                pool["particle_energy_GeV"] = ceh[cpw.I_EVTH_TOTAL_ENERGY_GEV]
                pool["num_photons"] = np.sum(light_field["size"])
                pool["num_bunches"] = num_bunches
                pool["num_all_bunches"] = int(num_all_bunches)

                c = lfc.parameterize_light_field(light_field=light_field)
                pool.update(c)
                pools.append(pool)

        return pools, explicit_steering


def make_cherenkov_pools_statistics(
    site,
    particle_id,
    particle_energy,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
    min_num_cherenkov_photons,
    density_cut,
    corsika_primary_path,
    run_id,
    prng,
):
    steering = make_steering(
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

    return estimate_cherenkov_pool(
        corsika_primary_steering=steering,
        corsika_primary_path=corsika_primary_path,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
        density_cut=density_cut,
    )


def init_light_field_from_corsika(bunches):
    lf = {}
    lf["x"] = bunches[:, cpw.IX] * cpw.CM2M  # cm to m
    lf["y"] = bunches[:, cpw.IY] * cpw.CM2M  # cm to m
    lf["cx"] = bunches[:, cpw.ICX]
    lf["cy"] = bunches[:, cpw.ICY]
    lf["t"] = bunches[:, cpw.ITIME] * 1e-9  # ns to s
    lf["size"] = bunches[:, cpw.IBSIZE]
    lf["wavelength"] = bunches[:, cpw.IWVL] * 1e-9  # nm to m
    return pandas.DataFrame(lf).to_records(index=False)


def extract_explicit_steering(corsika_run):
    """
    This can be used to reproduce the events bit by bit.
    """
    explicit_steering = {}
    explicit_steering["steering_card"] = str(corsika_run.steering_card)
    explicit_steering["primary_bytes"] = bytes(corsika_run.primary_bytes)
    return explicit_steering
