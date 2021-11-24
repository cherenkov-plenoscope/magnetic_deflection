import numpy as np
import os
import json_numpy
import pandas

from . import spherical_coordinates as sphcords
from . import tools


def parameterize_light_field(light_field):
    percentile = 50
    lf = light_field
    out = {}

    out["position_median_x_m"] = np.median(lf["x"])
    out["position_median_y_m"] = np.median(lf["y"])
    rel_x = lf["x"] - out["position_median_x_m"]
    rel_y = lf["y"] - out["position_median_y_m"]
    rel_r_square = rel_x ** 2 + rel_y ** 2
    del(rel_x)
    del(rel_y)
    rel_r_pivot = np.sqrt(np.percentile(a=rel_r_square, q=percentile))
    del(rel_r_square)
    out["position_radius50_m"] = rel_r_pivot


    out["direction_median_cx_rad"] = np.median(lf["cx"])
    out["direction_median_cy_rad"] = np.median(lf["cy"])
    direction_median_cz_rad = np.sqrt(
        1.0
        - out["direction_median_cx_rad"] ** 2
        - out["direction_median_cy_rad"] ** 2
    )

    direction_median = np.array([
        out["direction_median_cx_rad"],
        out["direction_median_cy_rad"],
        direction_median_cz_rad
    ])
    cz = np.sqrt(1.0 - lf["cx"] ** 2 - lf["cy"] ** 2)

    dot_product = np.array([
        lf["cx"] * direction_median[0],
        lf["cy"] * direction_median[1],
        cz * direction_median[2],
    ])
    cos_theta = np.sum(dot_product, axis=1)
    del(dot_product)
    cos_theta_pivot = np.percentile(a=cos_theta, q=percentile)
    del(cos_theta)
    theta_pivot = np.arccos(cos_theta_pivot)
    out["direction_angle50_rad"] = theta_pivot

    out["arrival_time_mean_s"] = float(np.mean(lf["t"]))
    out["arrival_time_median_s"] = float(np.median(lf["t"]))
    out["arrival_time_std_s"] = float(np.std(lf["t"]))
    return out


def inspect_pools(
    cherenkov_pools,
    off_axis_pivot_deg,
    instrument_azimuth_deg,
    instrument_zenith_deg,
):
    cers = pandas.DataFrame(cherenkov_pools)
    cers = cers.to_records(index=False)

    cer_azimuth_deg, cer_zenith_deg = sphcords._cx_cy_to_az_zd_deg(
        cx=cers["direction_median_cx_rad"], cy=cers["direction_median_cy_rad"],
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
    w_nph = make_weights_num_photons(num_photons=cers["num_photons"])

    weights = w_off * w_nph

    weights = limit_strong_weights(weights=weights, max_relative_weight=0.5)

    out = {}
    out["off_axis_deg"] = np.average(cer_off_axis_deg, weights=weights)

    for pkey in cers.dtype.names:
        _avg, _std = tools.average_std(cers[pkey], weights=weights)
        out[pkey] = _avg
        out[pkey + "_std"] = _std

    out["total_num_photons"] = np.sum(cers["num_photons"])
    out["total_num_showers"] = len(cers)

    if False:
        print("---")
        # print(json_numpy.dumps(out, indent=4))
        asw = np.argsort(weights)
        for i in range(len(asw)):
            j = asw[len(asw) - 1 - i]
            w_percent = int(100 * weights[j] / np.sum(weights))
            if w_percent > 1:
                print(
                    "weight {:03d} %, off-axis {:.2f} deg".format(
                        w_percent, cer_off_axis_deg[j],
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
