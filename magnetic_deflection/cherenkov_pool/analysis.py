import corsika_primary as cpw
from .. import spherical_coordinates
import numpy as np


def acos_accepting_numeric_tolerance(x, eps=1e-6):
    # input dimensionality
    x = np.asarray(x)
    scalar_input = False
    if x.ndim == 0:
        x = x[np.newaxis]  # Makes x 1D
        scalar_input = True

    # work
    assert eps >= 0.0
    mask = np.logical_and(x > 1.0, x < (1.0 + eps))
    x[mask] = 1.0
    mask = np.logical_and(x < -1.0, x > (-1.0 - eps))
    x[mask] = -1.0
    ret = np.arccos(x)

    # output dimensionality
    if scalar_input:
        return np.squeeze(ret)
    return ret


def init(light_field):
    percentile = np.percentile
    sqrt = np.sqrt
    median = np.median

    lf = light_field
    pool = {}

    if len(light_field["x"]) == 0:
        keys = [
            "cherenkov_x_m",
            "cherenkov_y_m",
            "cherenkov_radius50_m",
            "cherenkov_radius90_m",
            "cherenkov_cx_rad",
            "cherenkov_cy_rad",
            "cherenkov_half_angle50_rad",
            "cherenkov_half_angle90_rad",
            "cherenkov_t_s",
            "cherenkov_duration50_s",
            "cherenkov_duration90_s",
        ]
        for key in keys:
            pool[key] = float("nan")
        return pool

    pool["cherenkov_x_m"] = median(lf["x"])
    pool["cherenkov_y_m"] = median(lf["y"])
    pool["cherenkov_cx_rad"] = median(lf["cx"])
    pool["cherenkov_cy_rad"] = median(lf["cy"])
    pool["cherenkov_t_s"] = median(lf["t"])

    pool_radius = make_radius_wrt_center_position(
        photon_x_m=lf["x"],
        photon_y_m=lf["y"],
        center_x_m=pool["cherenkov_x_m"],
        center_y_m=pool["cherenkov_y_m"],
    )
    pool["cherenkov_radius50_m"] = percentile(a=pool_radius, q=50)
    pool["cherenkov_radius90_m"] = percentile(a=pool_radius, q=90)

    pool_theta = make_theta_wrt_center_direction(
        photon_cx_rad=lf["cx"],
        photon_cy_rad=lf["cy"],
        center_cx_rad=pool["cherenkov_cx_rad"],
        center_cy_rad=pool["cherenkov_cy_rad"],
    )
    pool["cherenkov_half_angle50_rad"] = percentile(a=pool_theta, q=50)
    pool["cherenkov_half_angle90_rad"] = percentile(a=pool_theta, q=90)

    time_delta_s = np.abs(lf["t"] - pool["cherenkov_t_s"])
    pool["cherenkov_duration50_s"] = 2.0 * percentile(a=time_delta_s, q=50)
    pool["cherenkov_duration90_s"] = 2.0 * percentile(a=time_delta_s, q=90)
    return pool


def make_radius_wrt_center_position(
    photon_x_m,
    photon_y_m,
    center_x_m,
    center_y_m,
):
    assert len(photon_x_m) == len(photon_y_m)
    rel_x = photon_x_m - center_x_m
    rel_y = photon_y_m - center_y_m
    return np.hypot(rel_x, rel_y)


def make_theta_wrt_center_direction(
    photon_cx_rad,
    photon_cy_rad,
    center_cx_rad,
    center_cy_rad,
):
    assert len(photon_cx_rad) == len(photon_cy_rad)

    center_cz_rad = spherical_coordinates.restore_cz(
        cx=center_cx_rad, cy=center_cy_rad
    )
    photon_cz_rad = spherical_coordinates.restore_cz(
        cx=photon_cx_rad, cy=photon_cy_rad
    )

    dot_product = np.array(
        [
            photon_cx_rad * center_cx_rad,
            photon_cy_rad * center_cy_rad,
            photon_cz_rad * center_cz_rad,
        ]
    )
    theta = acos_accepting_numeric_tolerance(
        np.sum(dot_product, axis=0),
        eps=1e-6,
    )
    del dot_product
    assert len(theta) == len(photon_cx_rad)
    return theta


def init_ellipse(x, y):
    median_x = np.median(x)
    median_y = np.median(y)

    cov_matrix = np.cov(np.c_[x, y].T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_values)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0

    major_axis = eigen_vectors[:, major_idx]
    major_std = np.sqrt(np.abs(eigen_values[major_idx]))
    minor_std = np.sqrt(np.abs(eigen_values[minor_idx]))

    azimuth = np.arctan2(major_axis[0], major_axis[1])
    return {
        "median_cx": median_cx,
        "median_cy": median_cy,
        "azimuth": azimuth,
        "major_std": major_std,
        "minor_std": minor_std,
    }
