import numpy as np
import os
import json_utils
import pandas

from . import spherical_coordinates as sphcords
from . import tools


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


def add_median_x_y_to_light_field(light_field):
    lf = light_field
    lf["median_x"] = np.median(lf["x"])
    lf["median_y"] = np.median(lf["y"])
    return lf


def add_median_cx_cy_to_light_field(light_field):
    lf = light_field
    lf["median_cx"] = np.median(lf["cx"])
    lf["median_cy"] = np.median(lf["cy"])
    return lf


def add_r_square_to_light_field_wrt_median(light_field):
    lf = light_field
    lf["r_square"] = make_radius_square_wrt_center_position(
        photon_x_m=lf["x"],
        photon_y_m=lf["y"],
        center_x_m=lf["median_x"],
        center_y_m=lf["median_y"],
    )
    return lf


def add_cos_theta_to_light_field_wrt_median(light_field):
    lf = light_field
    lf["cos_theta"] = make_cos_theta_wrt_center_direction(
        photon_cx_rad=lf["cx"],
        photon_cy_rad=lf["cy"],
        center_cx_rad=lf["median_cx"],
        center_cy_rad=lf["median_cy"],
    )
    return lf


def parameterize_light_field(light_field):
    percentile = 50
    lf = light_field
    out = {}

    out["cherenkov_x_m"] = lf["median_x"]
    out["cherenkov_y_m"] = lf["median_y"]

    r_pivot = np.sqrt(np.percentile(a=lf["r_square"], q=percentile))
    out["cherenkov_radius50_m"] = r_pivot

    out["cherenkov_cx_rad"] = lf["median_cx"]
    out["cherenkov_cy_rad"] = lf["median_cy"]

    cos_theta_pivot = np.percentile(a=lf["cos_theta"], q=percentile)
    out["cherenkov_angle50_rad"] = np.arccos(cos_theta_pivot)

    out["cherenkov_t_s"] = np.median(lf["t"])
    out["cherenkov_t_std_s"] = np.std(lf["t"])
    return out


def histogram_r_in_light_field(light_field, r_bin_edges):
    lf = light_field
    out = {}

    num_bins_r = len(r_bin_edges) - 1
    r_square_bin_edges = r_bin_edges**2
    hr = np.histogram(
        lf["r_square"], bins=r_square_bin_edges, weights=lf["size"]
    )[0]
    for iir in range(num_bins_r):
        iir_key = "cherenkov_r_bin_{:06d}".format(iir)
        out[iir_key] = hr[iir]
    return out


def histogram_theta_in_light_field(light_field, theta_bin_edges):
    lf = light_field
    out = {}
    num_bins_theta = len(theta_bin_edges) - 1
    theta = np.arccos(lf["cos_theta"])
    htheta = np.histogram(theta, bins=theta_bin_edges, weights=lf["size"])[0]
    for iit in range(num_bins_theta):
        iit_key = "cherenkov_theta_bin_{:06d}".format(iit)
        out[iit_key] = htheta[iit]
    return out


def inspect_pools(
    cherenkov_pools,
    off_axis_pivot_deg,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    ignore_bin_keys="_bin_",
):
    cers = pandas.DataFrame(cherenkov_pools)
    cers = cers.to_records(index=False)

    cer_azimuth_deg, cer_zenith_deg = sphcords._cx_cy_to_az_zd_deg(
        cx=cers["cherenkov_cx_rad"],
        cy=cers["cherenkov_cy_rad"],
    )

    cer_off_axis_deg = sphcords._angle_between_az_zd_deg(
        az1_deg=cer_azimuth_deg,
        zd1_deg=cer_zenith_deg,
        az2_deg=instrument_azimuth_deg,
        zd2_deg=instrument_zenith_deg,
    )

    w_off = make_weights_off_axis(
        off_axis_deg=cer_off_axis_deg, off_axis_pivot_deg=off_axis_pivot_deg
    )
    w_nph = make_weights_num_photons(num_photons=cers["cherenkov_num_photons"])

    weights = w_off * w_nph

    weights = limit_strong_weights(weights=weights, max_relative_weight=0.5)

    out = {}
    out["off_axis_deg"] = np.average(cer_off_axis_deg, weights=weights)

    for pkey in cers.dtype.names:
        if not ignore_bin_keys in pkey:
            _avg, _std = tools.average_std(cers[pkey], weights=weights)
            out[pkey] = _avg
            out[pkey + "_std"] = _std

    if False:
        print("---")
        # print(json_utils.dumps(out, indent=4))
        asw = np.argsort(weights)
        for i in range(len(asw)):
            j = asw[len(asw) - 1 - i]
            w_percent = int(100 * weights[j] / np.sum(weights))
            if w_percent > 1:
                print(
                    "weight {:03d} %, off-axis {:.2f} deg".format(
                        w_percent,
                        cer_off_axis_deg[j],
                    )
                )

    return out


def make_weights_num_photons(num_photons):
    """
    Returns weights (0 - 1]
    Reduce the weight of dim showers with only a few photons.
    These might create strong outliers.
    """
    med = np.median(num_photons)
    weights = num_photons / med
    weights[weights > 1.0] = 1.0
    return weights


def make_weights_off_axis(off_axis_deg, off_axis_pivot_deg):
    return np.exp(-0.5 * (off_axis_deg / off_axis_pivot_deg) ** 2)


def limit_strong_weights(weights, max_relative_weight):
    w = np.array(weights)
    w_relative = w / np.sum(w)

    while np.max(w_relative) > max_relative_weight:
        high = w_relative >= max_relative_weight
        w_relative[high] *= 0.95
        w_relative = w_relative / np.sum(w_relative)

    return w_relative
