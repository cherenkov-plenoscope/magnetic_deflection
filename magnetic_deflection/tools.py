import numpy as np
import pandas
import shutil
import os
import glob
import json_numpy
import corsika_primary_wrapper as cpw
import io
import tarfile

from . import spherical_coordinates
from . import Records
from . import recarray_io


def sort_records_by_key(records, keys):
    """
    Returns the records sorted by the values pointed to by keys

    Parameters
    ----------
    records : list of dicts
        A list of dicts where each dict has the entry pointed to by keys.
    keys : tuple of keys
        A tuple of keys to access each layer of the dict.
    """
    values = []
    for obj in records:
        val = obj
        for key in keys:
            val = val[key]
        values.append(val)
    order = np.argsort(values)
    return [records[order[i]] for i in range(len(records))]


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


def average_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def read_csv_to_dict(path):
    return pandas.read_csv(path).to_dict(orient="list")


def write_json(path, obj, indent=0):
    with open(path + ".tmp", "wt") as f:
        f.write(json_numpy.dumps(obj, indent=indent))
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


def append_jsonl_unsave(path, obj):
    with open(path, "at") as f:
        f.write(json_numpy.dumps(obj, indent=None))
        f.write("\n")


def read_jsonl(path):
    list_of_obj = []
    with open(path, "rt") as f:
        for line in f.readlines():
            obj = json_numpy.loads(line)
            list_of_obj.append(obj)
    return list_of_obj


def read_statistics_site_particle(map_site_particle_dir):
    map_dir = map_site_particle_dir
    paths = glob.glob(os.path.join(map_dir, "*_statistics.recarray.tar"))
    basenames = [os.path.basename(p) for p in paths]
    job_ids = [int(b[0:6]) for b in basenames]
    job_ids.sort()

    stats = Records.init(
        dtypes={
            "particle_azimuth_deg": "f4",
            "particle_zenith_deg":"f4",
            "num_photons": "f4",
            "num_bunches": "i4",
            "position_med_x_m": "f4",
            "position_med_y_m": "f4",
            "position_phi_rad": "f4",
            "position_std_minor_m": "f4",
            "position_std_major_m": "f4",
            "direction_med_cx_rad": "f4",
            "direction_med_cy_rad": "f4",
            "direction_phi_rad": "f4",
            "direction_std_minor_rad": "f4",
            "direction_std_major_rad": "f4",
            "arrival_time_mean_s": "f4",
            "arrival_time_median_s": "f4",
            "arrival_time_std_s": "f4",
            "off_axis_deg": "f4",
        }
    )

    num_jobs = len(job_ids)
    for i, job_id in enumerate(job_ids):
        print("Read job: {:06d}, {: 6d} / {: 6d}".format(job_id, i, num_jobs))
        s_path = os.path.join(
            map_dir, "{:06d}_statistics.recarray.tar".format(job_id)
        )
        pools = recarray_io.read_from_tar(path=s_path)
        stats = Records.append_numpy_recarray(stats, pool)

    return Records.to_numpy_recarray(stats)
