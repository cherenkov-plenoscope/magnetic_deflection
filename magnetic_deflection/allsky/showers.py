import os
import numpy as np
import rename_after_writing as rnw


def dtype():
    return [
        ("run", "u4"),
        ("event", "u4"),
        ("particle_azimuth_deg", "f4"),
        ("particle_zenith_deg", "f4"),
        ("particle_energy_GeV", "f4"),
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_num_bunches", "f4"),
        ("cherenkov_x_m", "f4"),
        ("cherenkov_y_m", "f4"),
        ("cherenkov_radius50_m", "f4"),
        ("cherenkov_cx_rad", "f4"),
        ("cherenkov_cy_rad", "f4"),
        ("cherenkov_angle50_rad", "f4"),
        ("cherenkov_t_s", "f4"),
        ("cherenkov_t_std_s", "f4"),
    ]


def write(path, showers):
    assert showers.dtype == dtype()
    with rnw.open(path, "wb") as f:
        f.write(showers.tobytes())


def read(path):
    with open(path, "rb") as f:
        showers = np.fromstring(f.read(), dtype=dtype())
    return showers


def num_records_in_file(path):
    stat = os.stat(path)
    size_in_bytes = stat.st_size
    return size_in_bytes // size_of_record_in_bytes()


def init(size=0):
    return np.core.records.recarray(
        shape=size,
        dtype=dtype(),
    )


def size_of_record_in_bytes():
    rr = np.core.records.recarray(
        shape=1,
        dtype=dtype(),
    )
    return len(rr.tobytes())
