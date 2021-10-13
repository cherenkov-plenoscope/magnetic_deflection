import pandas
import os
import numpy as np
import corsika_primary_wrapper as cpw
from . import corsika
from . import examples
from . import discovery
from . import light_field_characterization
from . import spherical_coordinates as sphcords
from . import tools


def make_jobs(
    work_dir,
    sites,
    particles,
    plenoscope_pointing,
    max_energy,
    num_energy_supports,
    energy_supports_power_law_slope=-1.7,
    initial_num_events_per_iteration=2 ** 5,
    max_total_num_events=2 ** 12,
    outlier_percentile=50.0,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
):
    abs_work_dir = os.path.abspath(work_dir)
    jobs = []
    for site_key in sites:
        for particle_key in particles:
            site = sites[site_key]
            particle = particles[particle_key]
            min_energy = np.min(
                particles[particle_key]["energy_bin_edges_GeV"]
            )
            energy_supports = tools.powerspace(
                start=min_energy,
                stop=max_energy,
                power_index=energy_supports_power_law_slope,
                num=num_energy_supports,
            )
            for energy_idx in range(len(energy_supports)):
                job = {}
                job["job"] = {}
                job["job"]["id"] = len(jobs)
                job["job"]["map_dir"] = os.path.join(abs_work_dir, "map")
                job["job"]["corsika_primary_path"] = corsika_primary_path

                job["site"] = dict(site)
                assert "key" not in job["site"]
                job["site"]["key"] = site_key

                job["particle"] = {}
                job["particle"]["key"] = particle_key
                job["particle"]["energy_GeV"] = energy_supports[energy_idx]
                job["particle"]["corsika_id"] = particle["particle_id"]

                job["instrument"] = plenoscope_pointing

                job["discovery"] = {
                    "outlier_percentile": 100.0,
                    "initial_num_events_per_iteration": initial_num_events_per_iteration,
                    "max_num_events": max_total_num_events,
                    "max_off_axis_deg": particle[
                        "magnetic_deflection_max_off_axis_deg"
                    ],
                    "min_num_cherenkov_photons": 1e2,
                }

                num_events = int(np.ceil(1e3 / job["particle"]["energy_GeV"]))
                num_events = np.min([1000, num_events])

                job["statistics"] = {
                    "num_events": num_events,
                    "outlier_percentile": outlier_percentile,
                    "off_axis_deg": 2.0 * particle[
                        "magnetic_deflection_max_off_axis_deg"
                    ],
                    "min_num_cherenkov_photons": 1e2,
                }

                jobs.append(job)

    return tools.sort_jobs_by_key(jobs=jobs, keys=("particle", "energy_GeV"))


def run_job(job):
    os.makedirs(job["job"]["map_dir"], exist_ok=True)
    prng = np.random.Generator(np.random.MT19937(seed=job["job"]["id"]))

    job_filename = "{:06d}_job.json".format(job["job"]["id"])
    discovery_filename = "{:06d}_discovery.jsonl".format(job["job"]["id"])
    statistics_filename = "{:06d}_statistics.jsonl".format(job["job"]["id"])
    result_filename = "{:06d}_result.json".format(job["job"]["id"])

    job_path = os.path.join(job["job"]["map_dir"], job_filename)
    discovery_path = os.path.join(job["job"]["map_dir"], discovery_filename)
    statistics_path = os.path.join(job["job"]["map_dir"], statistics_filename)
    result_path = os.path.join(job["job"]["map_dir"], result_filename)

    tools.write_json(job_path, job, indent=4)

    if not os.path.exists(discovery_path):
        guesses = discovery.estimate_deflection(
            prng=prng,
            site=job["site"],
            primary_energy=job["particle"]["energy_GeV"],
            primary_particle_id=job["particle"]["corsika_id"],
            instrument_azimuth_deg=job["instrument"]["azimuth_deg"],
            instrument_zenith_deg=job["instrument"]["zenith_deg"],
            max_off_axis_deg=job["discovery"]["max_off_axis_deg"],
            outlier_percentile=job["discovery"]["outlier_percentile"],
            initial_num_events_per_iteration=job["discovery"][
                "initial_num_events_per_iteration"
            ],
            min_num_cherenkov_photons=job["discovery"]["min_num_cherenkov_photons"],
            max_total_num_events=job["discovery"]["max_num_events"],
            corsika_primary_path=job["job"]["corsika_primary_path"],
            DEBUG_PRINT=True,
        )

        tools.write_jsonl(discovery_path, guesses)
    else:
        guesses = tools.read_jsonl(discovery_path)

    deflection = guesses[-1]

    if deflection["valid"]:

        if not os.path.exists(statistics_path):

            steering = corsika.make_steering(
                run_id=1 + job["job"]["id"],
                site=job["site"],
                primary_particle_id=job["particle"]["corsika_id"],
                primary_energy=job["particle"]["energy_GeV"],
                primary_cone_azimuth_deg=deflection["primary_azimuth_deg"],
                primary_cone_zenith_deg=deflection["primary_zenith_deg"],
                primary_cone_opening_angle_deg=job["statistics"]["off_axis_deg"],
                num_events=job["statistics"]["num_events"],
                prng=prng,
            )

            pools = corsika.estimate_cherenkov_pool(
                corsika_primary_steering=steering,
                corsika_primary_path=job["job"]["corsika_primary_path"],
                min_num_cherenkov_photons=job["statistics"]["min_num_cherenkov_photons"],
                outlier_percentile=job["statistics"]["outlier_percentile"],
            )
            tools.write_jsonl(statistics_path, pools)
        else:
            pools = tools.read_jsonl(statistics_path)

        assert len(pools) > 0

        pools = pandas.DataFrame(pools)
        pools = pools.to_records(index=False)

        (
            cherenkov_pool_azimuth_deg,
            cherenkov_pool_zenith_deg
        ) = sphcords._cx_cy_to_az_zd_deg(
            cx=pools["direction_median_cx_rad"],
            cy=pools["direction_median_cy_rad"],
        )

        delta_c_deg = sphcords._angle_between_az_zd_deg(
            az1_deg=cherenkov_pool_azimuth_deg,
            zd1_deg=cherenkov_pool_zenith_deg,
            az2_deg=job["instrument"]["azimuth_deg"],
            zd2_deg=job["instrument"]["zenith_deg"],
        )

        c_ref_deg = (1/2) * (job["statistics"]["off_axis_deg"])
        weights = np.exp(-0.5 * (delta_c_deg / c_ref_deg ) ** 2)
        weights = weights / np.sum(weights)

        out = {
            "char_position_med_x_m": np.average(pools["position_median_x_m"], weights=weights),
            "char_position_med_y_m": np.average(pools["position_median_y_m"], weights=weights),
            "char_position_phi_rad": np.average(pools["position_phi_rad"], weights=weights),
            "char_position_std_major_m": np.average(pools["position_std_major_m"], weights=weights),
            "char_position_std_minor_m": np.average(pools["position_std_minor_m"], weights=weights),

            "char_direction_med_cx_rad": np.average(pools["direction_median_cx_rad"], weights=weights),
            "char_direction_med_cy_rad": np.average(pools["direction_median_cy_rad"], weights=weights),
            "char_direction_phi_rad": np.average(pools["direction_phi_rad"], weights=weights),
            "char_direction_std_major_rad": np.average(pools["direction_std_major_rad"], weights=weights),
            "char_direction_std_minor_rad": np.average(pools["direction_std_minor_rad"], weights=weights),

            "char_arrival_time_mean_s": np.average(pools["arrival_time_mean_s"], weights=weights),
            "char_arrival_time_median_s": np.average(pools["arrival_time_median_s"], weights=weights),
            "char_arrival_time_std_s": np.average(pools["arrival_time_std_s"], weights=weights),
            "char_total_num_photons": np.sum(weights * pools["num_photons"]),
            "char_total_num_airshowers": len(pools),
            "char_outlier_percentile": job["statistics"]["outlier_percentile"],
        }

        deflection.update(out)
        tools.write_json(result_path, deflection)
    return 0


KEEP_KEYS = [
    "particle_id",
    "energy_GeV",
    "primary_azimuth_deg",
    "primary_zenith_deg",
    "cherenkov_pool_x_m",
    "cherenkov_pool_y_m",
    "off_axis_deg",
    "num_valid_Cherenkov_pools",
    "num_thrown_Cherenkov_pools",
    "total_num_events",
]


def structure_combined_results(
    combined_results, particles, sites,
):
    valid_results = []
    for result in combined_results:
        if result["valid"]:
            valid_results.append(result)

    df = pandas.DataFrame(valid_results)

    all_keys_keep = KEEP_KEYS + light_field_characterization.KEYS

    res = {}
    for site_key in sites:
        res[site_key] = {}
        for particle_key in particles:
            site_mask = (df["site_key"] == site_key).values
            particle_mask = (df["particle_key"] == particle_key).values
            mask = np.logical_and(site_mask, particle_mask)
            site_particle_df = df[mask]
            site_particle_df = site_particle_df[site_particle_df["valid"]]
            site_particle_keep_df = site_particle_df[all_keys_keep]
            site_particle_rec = site_particle_keep_df.to_records(index=False)
            argsort = np.argsort(site_particle_rec["energy_GeV"])
            site_particle_rec = site_particle_rec[argsort]
            res[site_key][particle_key] = site_particle_rec
    return res
