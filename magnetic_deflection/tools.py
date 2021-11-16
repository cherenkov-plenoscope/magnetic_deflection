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


def read_jsonl(path):
    list_of_obj = []
    with open(path, "rt") as f:
        for line in f.readlines():
            obj = json_numpy.loads(line)
            list_of_obj.append(obj)
    return list_of_obj
