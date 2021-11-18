import numpy as np
import os
import json_numpy
import pandas
import scipy
from scipy.spatial import KDTree
import time

from . import spherical_coordinates as sphcords
from . import tools


def estimate_ellipse(a, b):
    cov_matrix = np.cov(np.c_[a, b].T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_vals)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0
    major_axis = eigen_vecs[:, major_idx]
    phi = np.arctan2(major_axis[1], major_axis[0])
    major_std = np.sqrt(eigen_vals[major_idx])
    minor_std = np.sqrt(eigen_vals[minor_idx])

    return {
        "median_a": np.median(a),
        "median_b": np.median(b),
        "phi_rad": phi,
        "major_std": major_std,
        "minor_std": minor_std,
    }


def percentile_indices(values, target_value, percentile):
    """
    Rejecting outliers.
    Return a mask indicating the percentile of elements which are
    closest to the target_value.
    """
    assert 0.0 <= percentile <= 100.0
    factor = percentile / 100.0
    delta = np.abs(values - target_value)
    argsort_delta = np.argsort(delta)
    num_values = len(values)
    idxs = np.arange(num_values)
    idxs_sorted = idxs[argsort_delta]
    idx_limit = int(np.ceil(num_values * factor))
    valid_indices = idxs_sorted[0:idx_limit]
    mask = np.zeros(values.shape[0], dtype=np.bool)
    mask[valid_indices] = True
    return mask


def percentile_indices_wrt_median(values, percentile):
    """
    Rejecting outliers.
    Return a mask indicating the percentile of values which are
    closest to median(values)
    """
    return percentile_indices(
        values=values, target_value=np.median(values), percentile=percentile
    )


def light_field_density_cut(light_field, density_cut):
    if density_cut is None:
        return np.ones(len(light_field["x"]), dtype=np.int32)

    keys = list(density_cut.keys())
    assert len(keys) == 1
    key = keys[0]
    config = density_cut[key]

    if key == "median":
        return mask_inlier_in_light_field_geometry(
            light_field=light_field,
            percentile=config["percentile"],
        )
    elif key == "num_neighbors":
        num_neighbors = estimate_num_neighbors_in_light_field(
            light_field=light_field,
            xy_radius=config["xy_radius"],
            cxcy_radius=config["cxcy_radius"],
        )
        mask = num_neighbors >= config["min_num_neighbors"]
        return mask
    else:
        raise KeyError("Don't know density_cut '{:s}'.".format(key))


def mask_inlier_in_light_field_geometry(light_field, percentile):
    pc = percentile
    lf = light_field
    valid_x = percentile_indices_wrt_median(values=lf["x"], percentile=pc)
    valid_y = percentile_indices_wrt_median(values=lf["y"], percentile=pc)
    valid_cx = percentile_indices_wrt_median(values=lf["cx"], percentile=pc)
    valid_cy = percentile_indices_wrt_median(values=lf["cy"], percentile=pc)
    return np.logical_and(
        np.logical_and(valid_cx, valid_cy), np.logical_and(valid_x, valid_y)
    )


def parameterize_light_field(light_field):
    lf = light_field
    out = {}

    xy_ellipse = estimate_ellipse(a=lf["x"], b=lf["y"])
    out["position_med_x_m"] = xy_ellipse["median_a"]
    out["position_med_y_m"] = xy_ellipse["median_b"]
    out["position_phi_rad"] = xy_ellipse["phi_rad"]
    out["position_std_minor_m"] = xy_ellipse["minor_std"]
    out["position_std_major_m"] = xy_ellipse["major_std"]
    del xy_ellipse

    cxcy_ellipse = estimate_ellipse(a=lf["cx"], b=lf["cy"])
    out["direction_med_cx_rad"] = cxcy_ellipse["median_a"]
    out["direction_med_cy_rad"] = cxcy_ellipse["median_b"]
    out["direction_phi_rad"] = cxcy_ellipse["phi_rad"]
    out["direction_std_minor_rad"] = cxcy_ellipse["minor_std"]
    out["direction_std_major_rad"] = cxcy_ellipse["major_std"]
    del cxcy_ellipse

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
        cx=cers["direction_med_cx_rad"], cy=cers["direction_med_cy_rad"],
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


def estimate_num_neighbors_in_light_field(light_field, xy_radius, cxcy_radius):
    t0 = time.time()

    x = light_field["x"]
    y = light_field["y"]
    num = len(x)

    cx = light_field["cx"]
    cy = light_field["cy"]
    cz = np.sqrt(1 - cx**2 - cy**2)

    lf_xy = np.array(np.c_[x, y])
    lf_cxcycz = np.array(np.c_[cx, cy, cz])

    t1 = time.time()
    print("Init {:f}s".format(t1 - t0))

    tree_xy = scipy.spatial.cKDTree(data=lf_xy)
    tree_hemisphere = scipy.spatial.cKDTree(data=lf_cxcycz)

    t2 = time.time()
    print("Make trees {:f}s".format(t2 - t1))

    num_neighbors = np.zeros(num, dtype=np.uint32)

    for i in range(num):
        xy = np.c_[x[i], y[i]]
        near_xy_idx = tree_xy.query_ball_point(
            x=lf_xy[i],
            r=xy_radius
        )
        cxcy = np.c_[cx[i], cy[i]]
        near_he_idx = tree_hemisphere.query_ball_point(
            x=lf_cxcycz[i],
            r=cxcy_radius
        )

        near = set(near_xy_idx)
        near.update(near_he_idx)
        num_neighbors[i] = len(near)
    t3 = time.time()
    print("query trees {:f}s".format(t3 - t2))

    return num_neighbors

