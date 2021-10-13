import numpy as np
from . import discovery
import tempfile
import corsika_primary_wrapper as cpw
from . import spherical_coordinates as sphcords
import os
import pandas as pd


KEYPREFIX = "char_"

KEYS = [
    "char_position_med_x_m",
    "char_position_med_y_m",
    "char_position_phi_rad",
    "char_position_std_major_m",
    "char_position_std_minor_m",
    "char_direction_med_cx_rad",
    "char_direction_med_cy_rad",
    "char_direction_phi_rad",
    "char_direction_std_major_rad",
    "char_direction_std_minor_rad",
    "char_arrival_time_mean_s",
    "char_arrival_time_median_s",
    "char_arrival_time_std_s",
    "char_total_num_photons",
    "char_total_num_airshowers",
    "char_outlier_percentile",
]


def avgerage_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


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


def init_light_field_from_corsika(bunches):
    lf = {}
    lf["x"] = bunches[:, cpw.IX] * cpw.CM2M  # cm to m
    lf["y"] = bunches[:, cpw.IY] * cpw.CM2M  # cm to m
    lf["cx"] = bunches[:, cpw.ICX]
    lf["cy"] = bunches[:, cpw.ICY]
    lf["t"] = bunches[:, cpw.ITIME] * 1e-9  # ns to s
    lf["size"] = bunches[:, cpw.IBSIZE]
    lf["wavelength"] = bunches[:, cpw.IWVL] * 1e-9  # nm to m
    return pd.DataFrame(lf).to_records(index=False)


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
    out["position_median_x_m"] = xy_ellipse["median_a"]
    out["position_median_y_m"] = xy_ellipse["median_b"]
    out["position_phi_rad"] = xy_ellipse["phi_rad"]
    out["position_std_minor_m"] = xy_ellipse["minor_std"]
    out["position_std_major_m"] = xy_ellipse["major_std"]
    del xy_ellipse

    cxcy_ellipse = estimate_ellipse(a=lf["cx"], b=lf["cy"])
    out["direction_median_cx_rad"] = cxcy_ellipse["median_a"]
    out["direction_median_cy_rad"] = cxcy_ellipse["median_b"]
    out["direction_phi_rad"] = cxcy_ellipse["phi_rad"]
    out["direction_std_minor_rad"] = cxcy_ellipse["minor_std"]
    out["direction_std_major_rad"] = cxcy_ellipse["major_std"]
    del cxcy_ellipse

    out["arrival_time_mean_s"] = float(np.mean(lf["t"]))
    out["arrival_time_median_s"] = float(np.median(lf["t"]))
    out["arrival_time_std_s"] = float(np.std(lf["t"]))
    return out
