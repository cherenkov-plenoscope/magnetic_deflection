import corsika_primary as cpw
import numpy as np


def init(light_field):
    acos = np.arccos
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

    radius_square = make_radius_square_wrt_center_position(
        photon_x_m=lf["x"],
        photon_y_m=lf["y"],
        center_x_m=pool["cherenkov_x_m"],
        center_y_m=pool["cherenkov_y_m"],
    )
    pool["cherenkov_radius50_m"] = sqrt(percentile(a=radius_square, q=50))
    pool["cherenkov_radius90_m"] = sqrt(percentile(a=radius_square, q=90))

    cos_theta = make_cos_theta_wrt_center_direction(
        photon_cx_rad=lf["cx"],
        photon_cy_rad=lf["cy"],
        center_cx_rad=pool["cherenkov_cx_rad"],
        center_cy_rad=pool["cherenkov_cy_rad"],
    )
    pool["cherenkov_half_angle50_rad"] = acos(percentile(a=cos_theta, q=50))
    pool["cherenkov_half_angle90_rad"] = acos(percentile(a=cos_theta, q=90))

    time_delta_s = np.abs(lf["t"] - pool["cherenkov_t_s"])
    pool["cherenkov_duration50_s"] = 2.0 * percentile(a=time_delta_s, q=50)
    pool["cherenkov_duration90_s"] = 2.0 * percentile(a=time_delta_s, q=90)
    return pool


def make_radius_square_wrt_center_position(
    photon_x_m,
    photon_y_m,
    center_x_m,
    center_y_m,
):
    assert len(photon_x_m) == len(photon_y_m)
    rel_x = photon_x_m - center_x_m
    rel_y = photon_y_m - center_y_m
    return rel_x**2 + rel_y**2


def make_cos_theta_wrt_center_direction(
    photon_cx_rad,
    photon_cy_rad,
    center_cx_rad,
    center_cy_rad,
):
    assert len(photon_cx_rad) == len(photon_cy_rad)
    center_cz_rad = np.sqrt(1.0 - center_cx_rad**2 - center_cy_rad**2)
    center_direction = np.array([center_cx_rad, center_cy_rad, center_cz_rad])
    photon_cz_rad = np.sqrt(1.0 - photon_cx_rad**2 - photon_cy_rad**2)
    dot_product = np.array(
        [
            photon_cx_rad * center_direction[0],
            photon_cy_rad * center_direction[1],
            photon_cz_rad * center_direction[2],
        ]
    )
    cos_theta = np.sum(dot_product, axis=0)
    del dot_product
    assert len(cos_theta) == len(photon_cx_rad)
    return cos_theta
