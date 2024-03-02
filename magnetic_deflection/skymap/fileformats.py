import os
import numpy as np
import rename_after_writing as rnw
from . import recarray_utils


def pool_dtype():
    return [
        # identification
        ("run", "u4"),
        ("event", "u4"),
        # primary particle
        ("particle_cx", "f4"),
        ("particle_cy", "f4"),
        ("particle_energy_GeV", "f4"),
        # size
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_num_bunches", "f4"),
        # maximum
        ("cherenkov_altitude_p16_m", "f4"),
        ("cherenkov_altitude_p50_m", "f4"),
        ("cherenkov_altitude_p84_m", "f4"),
        # geometry
        ("cherenkov_x_p16_m", "f4"),
        ("cherenkov_x_p50_m", "f4"),
        ("cherenkov_x_p84_m", "f4"),
        ("cherenkov_y_p16_m", "f4"),
        ("cherenkov_y_p50_m", "f4"),
        ("cherenkov_y_p84_m", "f4"),
        ("cherenkov_x_modus_m", "f4"),
        ("cherenkov_y_modus_m", "f4"),
        ("cherenkov_cx_p16_rad", "f4"),
        ("cherenkov_cx_p50_rad", "f4"),
        ("cherenkov_cx_p84_rad", "f4"),
        ("cherenkov_cy_p16_rad", "f4"),
        ("cherenkov_cy_p50_rad", "f4"),
        ("cherenkov_cy_p84_rad", "f4"),
        ("cherenkov_cx_modus", "f4"),
        ("cherenkov_cy_modus", "f4"),
        # time
        ("cherenkov_time_p16_ns", "f4"),
        ("cherenkov_time_p50_ns", "f4"),
        ("cherenkov_time_p84_ns", "f4"),
    ]


def PoolUtils():
    return recarray_utils.RecarrayUtils(dtype=pool_dtype)
