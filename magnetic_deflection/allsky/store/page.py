import os
import numpy as np
import rename_after_writing as rnw


def dtype():
    return [
        # identification
        ("run", "u4"),
        ("event", "u4"),
        # primary particle
        ("particle_cx_rad", "f4"),
        ("particle_cy_rad", "f4"),
        ("particle_energy_GeV", "f4"),
        # size
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_num_bunches", "f4"),
        # maximum
        ("cherenkov_maximum_asl_m", "f4"),
        # geometry
        ("cherenkov_x_m", "f4"),
        ("cherenkov_y_m", "f4"),
        ("cherenkov_radius50_m", "f4"),
        ("cherenkov_radius90_m", "f4"),
        ("cherenkov_cx_rad", "f4"),
        ("cherenkov_cy_rad", "f4"),
        ("cherenkov_half_angle50_rad", "f4"),
        ("cherenkov_half_angle90_rad", "f4"),
        # time
        ("cherenkov_t_s", "f4"),
        ("cherenkov_duration50_s", "f4"),
        ("cherenkov_duration90_s", "f4"),
    ]


def write(path, page):
    assert page.dtype == dtype()
    with rnw.open(path, "wb") as f:
        f.write(page.tobytes())


def read(path):
    with open(path, "rb") as f:
        page = np.fromstring(f.read(), dtype=dtype())
    return page


def num_records_in_file(path):
    stat = os.stat(path)
    size_in_bytes = stat.st_size
    return size_in_bytes // size_of_record_in_bytes()


def init(size=0):
    return np.recarray(
        shape=size,
        dtype=dtype(),
    )


def size_of_record_in_bytes():
    rr = np.recarray(
        shape=1,
        dtype=dtype(),
    )
    return len(rr.tobytes())
