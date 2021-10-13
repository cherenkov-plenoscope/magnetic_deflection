import pandas
import shutil
import os
import glob
import numpy as np
import json
import json_numpy
import corsika_primary_wrapper as cpw
from . import corsika
from . import examples
from . import discovery
from . import light_field_characterization
from . import spherical_coordinates as sphcords


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
            particle_id = particles[particle_key]["particle_id"]
            min_energy = np.min(
                particles[particle_key]["energy_bin_edges_GeV"]
            )
            energy_supports = powerspace(
                start=min_energy,
                stop=max_energy,
                power_index=energy_supports_power_law_slope,
                num=num_energy_supports,
            )
            for energy_idx in range(len(energy_supports)):
                job = {}
                job["seed"] = len(jobs)
                job["map_dir"] = os.path.join(abs_work_dir, "map")
                job["site"] = site
                job["primary_energy"] = energy_supports[energy_idx]
                job["primary_particle_id"] = particle_id
                job["instrument_azimuth_deg"] = plenoscope_pointing[
                    "azimuth_deg"
                ]
                job["instrument_zenith_deg"] = plenoscope_pointing[
                    "zenith_deg"
                ]
                job["max_off_axis_deg"] = particles[particle_key][
                    "magnetic_deflection_max_off_axis_deg"
                ]
                job["corsika_primary_path"] = corsika_primary_path
                job["site_key"] = site_key
                job["particle_key"] = particle_key
                job[
                    "initial_num_events_per_iteration"
                ] = initial_num_events_per_iteration
                job["max_total_num_events"] = max_total_num_events
                job["outlier_percentile_discovery"] = 100.0
                job["outlier_percentile_statistics"] = outlier_percentile
                jobs.append(job)
    return sort_jobs_by_key(jobs=jobs, key="primary_energy")


def sort_jobs_by_key(jobs, key):
    _values = [job[key] for job in jobs]
    _values_argsort = np.argsort(_values)
    jobs_sorted = [jobs[_values_argsort[i]] for i in range(len(jobs))]
    return jobs_sorted


def run_job(job):
    os.makedirs(job["map_dir"], exist_ok=True)
    prng = np.random.Generator(np.random.MT19937(seed=job["seed"]))

    job_filename = "{:06d}_job.json".format(job["seed"])
    discovery_filename = "{:06d}.json".format(job["seed"])
    statistics_filename = "{:06d}_showers.jsonl".format(job["seed"])

    job_path
    discovery_path = os.path.join(job["map_dir"], discovery_filename)
    statistics_path = os.path.join(job["map_dir"], statistics_filename)

    write_json(, )

    if not os.path.exists(discovery_path):
        deflection = discovery.estimate_deflection(
            prng=prng,
            site=job["site"],
            primary_energy=job["primary_energy"],
            primary_particle_id=job["primary_particle_id"],
            instrument_azimuth_deg=job["instrument_azimuth_deg"],
            instrument_zenith_deg=job["instrument_zenith_deg"],
            max_off_axis_deg=job["max_off_axis_deg"],
            outlier_percentile=job["outlier_percentile_discovery"],
            initial_num_events_per_iteration=job[
                "initial_num_events_per_iteration"
            ],
            min_num_valid_Cherenkov_pools=5,
            min_num_cherenkov_photons=1e2,
            max_total_num_events=job["max_total_num_events"],
            corsika_primary_path=job["corsika_primary_path"],
            DEBUG_PRINT=True,
        )
        deflection["particle_id"] = job["primary_particle_id"]
        deflection["energy_GeV"] = job["primary_energy"]
        deflection["site_key"] = job["site_key"]
        deflection["particle_key"] = job["particle_key"]

        write_json(discovery_path, deflection)
    else:
        deflection = read_json(discovery_path)


    if deflection["valid"]:

        total_energy_thrown = 1e2
        num_events = int(np.ceil(total_energy_thrown / job["primary_energy"]))

        if not os.path.exists(statistics_path):

            steering = corsika.make_steering(
                run_id=job["seed"],
                site=job["site"],
                primary_particle_id=job["primary_particle_id"],
                primary_energy=job["primary_energy"],
                primary_cone_azimuth_deg=deflection["primary_azimuth_deg"],
                primary_cone_zenith_deg=deflection["primary_zenith_deg"],
                primary_cone_opening_angle_deg=job["max_off_axis_deg"],
                num_events=num_events,
                prng=prng,
            )

            pools = corsika.estimate_cherenkov_pool(
                corsika_primary_steering=steering,
                corsika_primary_path=job["corsika_primary_path"],
                min_num_cherenkov_photons=1e2,
                outlier_percentile=job["outlier_percentile_statistics"],
            )
            write_jsonl(statistics_path, pools)
        else:
            pools = read_jsonl(statistics_path)

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
                az2_deg=job["instrument_azimuth_deg"],
                zd2_deg=job["instrument_zenith_deg"],
            )

            c_ref_deg = (1/2) * (job["max_off_axis_deg"])
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
                "char_outlier_percentile": job["outlier_percentile_statistics"],
            }

            deflection.update(out)
            write_json(discovery_path, deflection)
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


def write_recarray_to_csv(recarray, path):
    df = pandas.DataFrame(recarray)
    csv = df.to_csv(index=False)
    with open(path + ".tmp", "wt") as f:
        f.write(csv)
    shutil.move(path + ".tmp", path)


def read_csv_to_recarray(path):
    df = pandas.read_csv(path)
    rec = df.to_records(index=False)
    return rec


def read_deflection_table(path):
    paths = glob.glob(os.path.join(path, "*.csv"))
    deflection_table = {}
    for pa in paths:
        basename = os.path.basename(pa)
        name = basename.split(".")[0]
        split_name = name.split("_")
        assert len(split_name) == 2
        site_key, particle_key = split_name
        if site_key not in deflection_table:
            deflection_table[site_key] = {}
        deflection_table[site_key][particle_key] = read_csv_to_recarray(pa)
    return deflection_table


def write_deflection_table(deflection_table, path):
    for site_key in deflection_table:
        for particle_key in deflection_table[site_key]:
            out_path = os.path.join(
                path, "{:s}_{:s}.csv".format(site_key, particle_key)
            )
            write_recarray_to_csv(
                recarray=deflection_table[site_key][particle_key],
                path=out_path,
            )


def powerspace(start, stop, power_index, num, iterations=10000):
    assert num >= 2
    num_points_without_start_and_end = num - 2
    if num_points_without_start_and_end >= 1:
        full = []
        for iti in range(iterations):
            points = np.sort(
                cpw.random_distributions.draw_power_law(
                    prng=np.random.default_rng(),
                    lower_limit=start,
                    upper_limit=stop,
                    power_slope=power_index,
                    num_samples=num_points_without_start_and_end,
                )
            )
            points = [start] + points.tolist() + [stop]
            full.append(points)
        full = np.array(full)
        return np.mean(full, axis=0)
    else:
        return np.array([start, stop])


def read_csv_to_dict(path):
    return pandas.read_csv(path).to_dict(orient="list")


def write_json(path, obj):
    with open(path + ".tmp", "wt") as f:
        f.write(json_numpy.dumps(obj))
    shutil.move(path + ".tmp", path)


def read_json(path):
    with open(path, "rt") as f:
        obj = json_numpy.loads(f.read())
    return obj


def write_jsonl(path, list_of_obj):
    with open(path + ".tmp", "wt") as f:
        for obj in list_of_obj:
            f.write(json_numpy.dumps(obj, indent=None))
            f.write("\n")
    shutil.move(path + ".tmp", path)


def read_jsonl(path):
    list_of_obj = []
    with open(path, "rt") as f:
        for line in f.readlines():
            obj = json_numpy.loads(line)
            list_of_obj.append(obj)
    return list_of_obj
