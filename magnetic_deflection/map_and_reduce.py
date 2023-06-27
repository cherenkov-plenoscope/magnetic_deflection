import os
import numpy as np
import pandas
import json_line_logger as jlogging

from . import corsika
from . import examples
from . import discovery
from . import light_field_characterization
from . import tools
from . import spherical_coordinates
from . import recarray_io
from . import work_dir_structure


def make_jobs(
    first_job_id,
    work_dir,
    site,
    site_key,
    particle,
    particle_key,
    pointing,
    energy_supports_min,
    energy_supports_max,
    energy_supports_num,
    energy_supports_power_law_slope,
    discovery_max_total_energy,
    discovery_min_energy_per_iteration,
    discovery_min_num_showers_per_iteration,
    statistics_total_energy,
    statistics_min_num_showers,
    statistics_r_bin_edges,
    statistics_theta_bin_edges_deg,
    min_num_cherenkov_photons,
    corsika_primary_path,
):
    assert energy_supports_min > 0
    assert energy_supports_min < energy_supports_max
    assert energy_supports_max > 0
    assert energy_supports_num > 1

    assert min_num_cherenkov_photons > 0

    assert statistics_min_num_showers > 0
    assert statistics_total_energy >= energy_supports_max

    assert discovery_min_num_showers_per_iteration > 0
    assert discovery_min_energy_per_iteration > 0
    assert discovery_max_total_energy >= energy_supports_max

    abs_work_dir = os.path.abspath(work_dir)
    jobs = []

    energy_supports = tools.powerspace(
        start=energy_supports_min,
        stop=energy_supports_max,
        power_index=energy_supports_power_law_slope,
        num=energy_supports_num,
    )
    for energy_idx in range(len(energy_supports)):
        job = {}
        job["job"] = {}
        job["job"]["id"] = first_job_id + len(jobs)
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
        job["particle"]["corsika_id"] = particle["corsika_particle_id"]

        job["pointing"] = pointing

        max_total_num_showers = int(
            np.ceil(discovery_max_total_energy / job["particle"]["energy_GeV"])
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
            "num_showers_per_iteration": num_showers,
            "max_num_showers": max_total_num_showers,
            "max_off_axis_deg": particle["magnetic_deflection"][
                "max_acceptable_off_axis_angle_deg"
            ],
            "min_num_cherenkov_photons": min_num_cherenkov_photons,
        }

        num_showers = int(
            np.ceil(statistics_total_energy / job["particle"]["energy_GeV"])
        )

        num_showers = np.max([num_showers, statistics_min_num_showers])

        statistics_theta_bin_edges = np.deg2rad(statistics_theta_bin_edges_deg)
        job["statistics"] = {
            "num_showers": num_showers,
            "min_num_cherenkov_photons": min_num_cherenkov_photons,
            "off_axis_deg": 3.0
            * particle["magnetic_deflection"][
                "max_acceptable_off_axis_angle_deg"
            ],
            "optional": {
                "histogram_r": {
                    "r_bin_edges": np.array(statistics_r_bin_edges),
                },
                "histogram_theta": {
                    "theta_bin_edges": np.array(statistics_theta_bin_edges),
                },
            },
        }

        jobs.append(job)

    return jobs


def run_job(job):
    os.makedirs(job["job"]["map_dir"], exist_ok=True)
    map_paths = work_dir_structure.map_paths(
        map_dir=job["job"]["map_dir"], job_id=job["job"]["id"]
    )
    logger = jlogging.LoggerFile(path=map_paths["log"])
    logger.info("job: start")
    logger.info(
        "job: site: {:s}, particle: {:s}, energy: {:f} GeV".format(
            job["site"]["key"],
            job["particle"]["key"],
            job["particle"]["energy_GeV"],
        )
    )

    tools.write_json(map_paths["job"], job, indent=4)
    logger.info("job: init prng for discovery with seed=job_id")
    prng = np.random.Generator(np.random.MT19937(seed=job["job"]["id"]))

    logger.info("job: discovering deflection")
    if not os.path.exists(map_paths["discovery"]):
        logger.info("job: estimate new guess for deflection")
        estimates = discovery.estimate_deflection(
            logger=logger,
            prng=prng,
            site=job["site"],
            particle_energy=job["particle"]["energy_GeV"],
            particle_id=job["particle"]["corsika_id"],
            instrument_azimuth_deg=job["pointing"]["azimuth_deg"],
            instrument_zenith_deg=job["pointing"]["zenith_deg"],
            max_off_axis_deg=job["discovery"]["max_off_axis_deg"],
            num_showers_per_iteration=job["discovery"][
                "num_showers_per_iteration"
            ],
            max_num_showers=job["discovery"]["max_num_showers"],
            min_num_cherenkov_photons=job["discovery"][
                "min_num_cherenkov_photons"
            ],
            corsika_primary_path=job["job"]["corsika_primary_path"],
        )

        tools.write_jsonl(map_paths["discovery"], estimates)
    else:
        logger.info("job: use existing guess for deflection")
        estimates = tools.read_jsonl(map_paths["discovery"])

    if len(estimates) > 0:
        best_estimate = estimates[-1]
        logger.info("job: have valid estimate for deflection")

        if best_estimate["off_axis_deg"] > job["statistics"]["off_axis_deg"]:
            logger.info("job: increase opening angle to gather statistics.")
            cone_opening_angle_deg = best_estimate["off_axis_deg"]
        else:
            cone_opening_angle_deg = job["statistics"]["off_axis_deg"]

        if (
            best_estimate["particle_zenith_deg"] + cone_opening_angle_deg
            > corsika.MAX_ZENITH_DEG
        ):
            logger.info(
                "job: Warning: Cone's opening is out of valid zenith-range."
            )

        if (
            best_estimate["particle_zenith_deg"] - cone_opening_angle_deg
            > corsika.MAX_ZENITH_DEG
        ):
            logger.info(
                "job: Error: Cone is completely out of valid zenith-range."
            )
            return 0

        logger.info("job: gathering statistics")
        logger.info("job: re-init prng for statistics with seed=job_id")
        prng = np.random.Generator(np.random.MT19937(seed=job["job"]["id"]))

        if not os.path.exists(map_paths["statistics"]):
            logger.info("job: simulate new showers")

            (
                pools,
                pools_reproduction,
            ) = corsika.make_cherenkov_pools_statistics(
                site=job["site"],
                particle_id=job["particle"]["corsika_id"],
                particle_energy=job["particle"]["energy_GeV"],
                particle_cone_azimuth_deg=best_estimate[
                    "particle_azimuth_deg"
                ],
                particle_cone_zenith_deg=best_estimate["particle_zenith_deg"],
                particle_cone_opening_angle_deg=cone_opening_angle_deg,
                num_showers=job["statistics"]["num_showers"],
                min_num_cherenkov_photons=job["statistics"][
                    "min_num_cherenkov_photons"
                ],
                corsika_primary_path=job["job"]["corsika_primary_path"],
                run_id=job["job"]["id"],
                prng=prng,
                statistics_optional=job["statistics"]["optional"],
            )

            pools = add_off_axis_to_pool_statistics(
                pool_statistics=pools,
                instrument_azimuth_deg=job["pointing"]["azimuth_deg"],
                instrument_zenith_deg=job["pointing"]["zenith_deg"],
            )

            corsika.cpw.steering.write_steerings_and_seeds(
                path=map_paths["statistics_steering"],
                runs={job["job"]["id"]: pools_reproduction},
            )

            _write_pool_statistics(
                pool_statistics=pools, path=map_paths["statistics"]
            )
        else:
            logger.info("job: use existing showers")
            pools = _read_pool_statistics(path=map_paths["statistics"])

        logger.info("job: estimate statistics of showers")
        deflection = light_field_characterization.inspect_pools(
            cherenkov_pools=pools,
            off_axis_pivot_deg=(1 / 2) * (job["statistics"]["off_axis_deg"]),
            instrument_azimuth_deg=job["pointing"]["azimuth_deg"],
            instrument_zenith_deg=job["pointing"]["zenith_deg"],
        )
        deflection["particle_energy_GeV"] = job["particle"]["energy_GeV"]

        logger.info("job: write results")
        tools.write_json(map_paths["deflection"], deflection)

    logger.info("job: end")
    return 0


def add_off_axis_to_pool_statistics(
    pool_statistics, instrument_azimuth_deg, instrument_zenith_deg
):
    for i in range(len(pool_statistics)):
        cer_az_deg, cer_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=pool_statistics[i]["cherenkov_cx_rad"],
            cy=pool_statistics[i]["cherenkov_cy_rad"],
        )
        off_axis_deg = spherical_coordinates._angle_between_az_zd_deg(
            az1_deg=cer_az_deg,
            zd1_deg=cer_zd_deg,
            az2_deg=instrument_azimuth_deg,
            zd2_deg=instrument_zenith_deg,
        )
        pool_statistics[i]["off_axis_deg"] = float(off_axis_deg)
    return pool_statistics


def _write_pool_statistics(pool_statistics, path):
    pool_statistics_rec = pandas.DataFrame(pool_statistics).to_records(
        index=False
    )
    pool_statistics_rec = recarray_io.change_dtype(
        recarray=pool_statistics_rec, current_dtype="f8", target_dtype="f4"
    )
    pool_statistics_rec = recarray_io.change_dtype(
        recarray=pool_statistics_rec, current_dtype="i8", target_dtype="i4"
    )
    recarray_io.write_to_tar(recarray=pool_statistics_rec, path=path)


def _read_pool_statistics(path):
    pool_statistics_rec = recarray_io.read_from_tar(path=path)
    pool_statistics_df = pandas.DataFrame(pool_statistics_rec)
    del pool_statistics_rec
    pool_statistics = pool_statistics_df.to_dict(orient="records")
    del pool_statistics_df
    return pool_statistics
