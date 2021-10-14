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
from . import jsonl_logger


def make_jobs(
    work_dir,
    sites,
    particles,
    pointing,
    energy_supports_max,
    energy_supports_num,
    energy_supports_power_law_slope=-1.7,
    discovery_max_total_energy=4e3,
    discovery_min_energy_per_iteration=12.0,
    discovery_min_num_showers_per_iteration=16,
    statistics_total_energy=2e3,
    statistics_min_num_showers=16,
    outlier_percentile=50.0,
    min_num_cherenkov_photons=100,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
):
    assert discovery_max_total_energy > (
        discovery_min_num_showers_per_iteration * energy_supports_max
    )
    assert statistics_total_energy > (
        statistics_min_num_showers * energy_supports_max
    )

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
                stop=energy_supports_max,
                power_index=energy_supports_power_law_slope,
                num=energy_supports_num,
            )
            for energy_idx in range(len(energy_supports)):
                job = {}
                job["job"] = {}
                job["job"]["id"] = len(jobs)
                job["job"]["map_dir"] = os.path.join(
                    abs_work_dir, "map", site_key, particle_key
                )
                job["job"]["corsika_primary_path"] = corsika_primary_path

                job["site"] = dict(site)
                assert "key" not in job["site"]
                job["site"]["key"] = site_key

                job["particle"] = {}
                job["particle"]["key"] = particle_key
                job["particle"]["energy_GeV"] = energy_supports[energy_idx]
                job["particle"]["corsika_id"] = particle["particle_id"]

                job["pointing"] = pointing

                max_total_num_showers = int(
                    np.ceil(
                        discovery_max_total_energy
                        / job["particle"]["energy_GeV"]
                    )
                )
                num_showers = int(
                    np.ceil(
                        discovery_min_energy_per_iteration
                        / job["particle"]["energy_GeV"]
                    )
                )
                num_showers = np.max(
                    [num_showers, discovery_min_num_showers_per_iteration]
                )

                job["discovery"] = {
                    "outlier_percentile": 100.0,
                    "num_showers_per_iteration": num_showers,
                    "max_num_showers": max_total_num_showers,
                    "max_off_axis_deg": particle[
                        "magnetic_deflection_max_off_axis_deg"
                    ],
                    "min_num_cherenkov_photons": min_num_cherenkov_photons,
                }

                num_showers = int(
                    np.ceil(
                        statistics_total_energy / job["particle"]["energy_GeV"]
                    )
                )
                num_showers = np.min([1000, num_showers])

                job["statistics"] = {
                    "num_showers": num_showers,
                    "outlier_percentile": outlier_percentile,
                    "off_axis_deg": 2.0
                    * particle["magnetic_deflection_max_off_axis_deg"],
                    "min_num_cherenkov_photons": min_num_cherenkov_photons,
                }

                jobs.append(job)

    return tools.sort_records_by_key(
        records=jobs, keys=("particle", "energy_GeV")
    )


def run_job(job):

    os.makedirs(job["job"]["map_dir"], exist_ok=True)

    log_filename = "{:06d}_log.json".format(job["job"]["id"])
    log_path = os.path.join(job["job"]["map_dir"], log_filename)
    jlog = jsonl_logger.init(path=log_path)
    jlog.info("job: start")

    job_filename = "{:06d}_job.json".format(job["job"]["id"])
    job_path = os.path.join(job["job"]["map_dir"], job_filename)
    tools.write_json(job_path, job, indent=4)

    discovery_filename = "{:06d}_discovery.jsonl".format(job["job"]["id"])
    discovery_path = os.path.join(job["job"]["map_dir"], discovery_filename)

    statistics_filename = "{:06d}_statistics.jsonl".format(job["job"]["id"])
    statistics_path = os.path.join(job["job"]["map_dir"], statistics_filename)

    result_filename = "{:06d}_result.json".format(job["job"]["id"])
    result_path = os.path.join(job["job"]["map_dir"], result_filename)

    prng = np.random.Generator(np.random.MT19937(seed=job["job"]["id"]))

    jlog.info("job: start discovery of deflection")
    if not os.path.exists(discovery_path):
        jlog.info("job: estimate new guess for deflection")
        guesses = discovery.estimate_deflection(
            json_logger=jlog,
            prng=prng,
            site=job["site"],
            particle_energy=job["particle"]["energy_GeV"],
            particle_id=job["particle"]["corsika_id"],
            instrument_azimuth_deg=job["pointing"]["azimuth_deg"],
            instrument_zenith_deg=job["pointing"]["zenith_deg"],
            max_off_axis_deg=job["discovery"]["max_off_axis_deg"],
            outlier_percentile=job["discovery"]["outlier_percentile"],
            num_showers_per_iteration=job["discovery"][
                "num_showers_per_iteration"
            ],
            max_num_showers=job["discovery"]["max_num_showers"],
            min_num_cherenkov_photons=job["discovery"][
                "min_num_cherenkov_photons"
            ],
            corsika_primary_path=job["job"]["corsika_primary_path"],
            guesses_path=discovery_path,
        )

        tools.write_jsonl(discovery_path, guesses)
    else:
        jlog.info("job: use existing guess for deflection")
        guesses = tools.read_jsonl(discovery_path)

    guess = guesses[-1]

    if guess["valid"]:
        jlog.info("job: start gathering statistics of showers")

        if not os.path.exists(statistics_path):
            jlog.info("job: simulate new showers")

            steering = corsika.make_steering(
                run_id=1 + job["job"]["id"],
                site=job["site"],
                particle_id=job["particle"]["corsika_id"],
                particle_energy=job["particle"]["energy_GeV"],
                particle_cone_azimuth_deg=guess["particle_azimuth_deg"],
                particle_cone_zenith_deg=guess["particle_zenith_deg"],
                particle_cone_opening_angle_deg=job["statistics"][
                    "off_axis_deg"
                ],
                num_showers=job["statistics"]["num_showers"],
                prng=prng,
            )

            pools = corsika.estimate_cherenkov_pool(
                corsika_primary_steering=steering,
                corsika_primary_path=job["job"]["corsika_primary_path"],
                min_num_cherenkov_photons=job["statistics"][
                    "min_num_cherenkov_photons"
                ],
                outlier_percentile=job["statistics"]["outlier_percentile"],
            )
            tools.write_jsonl(statistics_path, pools)
        else:
            jlog.info("job: use existing showers")
            pools = tools.read_jsonl(statistics_path)

        assert len(pools) > 0

        jlog.info("job: estimate statistics of showers")
        out = light_field_characterization.inspect_pools(
            cherenkov_pools=pools,
            off_axis_pivot_deg=(1 / 2) * (job["statistics"]["off_axis_deg"]),
            instrument_azimuth_deg=job["pointing"]["azimuth_deg"],
            instrument_zenith_deg=job["pointing"]["zenith_deg"],
        )
        out["outlier_percentile"] = job["statistics"]["outlier_percentile"]

        jlog.info("job: write results")
        for okey in out:
            guess[light_field_characterization.KEYPREFIX + okey] = out[okey]
        tools.write_json(result_path, guess)

    jlog.info("job: end")
    return 0


KEEP_KEYS = [
    "particle_id",
    "energy_GeV",
    "particle_azimuth_deg",
    "particle_zenith_deg",
    "cherenkov_pool_x_m",
    "cherenkov_pool_y_m",
    "off_axis_deg",
    "num_valid_Cherenkov_pools",
    "num_thrown_Cherenkov_pools",
    "total_num_showers",
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
