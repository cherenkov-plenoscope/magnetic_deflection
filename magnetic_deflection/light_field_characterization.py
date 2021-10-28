import numpy as np
import os
import json_numpy
import pandas
import scipy
from scipy import spatial

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
    debug_print=False,
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
        off_axis_deg=cer_off_axis_deg,
        off_axis_pivot_deg=off_axis_pivot_deg
    )
    w_nph = make_weights_num_photons(num_photons=cers["num_photons"])

    w_off_nph = w_off * w_nph

    weights = make_neighborhood_weights(
        cer_cx_rad=cers["direction_med_cx_rad"],
        cer_cy_rad=cers["direction_med_cy_rad"],
        cer_weights=w_off_nph,
        pivot_radius_rad=4 * np.deg2rad(off_axis_pivot_deg),
        min_num_neighborhood=2
    )
    if np.sum(weights) == 0:
        print("fallback, no neighbor wights.")

    weights = w_off_nph

    out = {}
    out["off_axis_deg"] = np.average(cer_off_axis_deg, weights=weights)

    for pkey in cers.dtype.names:
        _avg, _std = tools.average_std(cers[pkey], weights=weights)
        out[pkey] = _avg
        out[pkey + "_std"] = _std

    out["total_num_photons"] = np.sum(cers["num_photons"])
    out["total_num_showers"] = len(cers)

    if True:
        print("---")
        #print(json_numpy.dumps(out, indent=4))
        asw = np.argsort(weights)
        for i in range(int(len(asw) / 10)):
            j = asw[len(asw) - 1 - i]
            print(
                "weight {:03d} %, off-axis {:.2f} deg".format(
                    int(100 * weights[j]/np.sum(weights)), cer_off_axis_deg[j],
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
    weights = (num_photons / med)
    weights[weights > 1.0] = 1.0
    return weights


def make_weights_off_axis(off_axis_deg, off_axis_pivot_deg):
    return np.exp(-0.5 * (off_axis_deg / off_axis_pivot_deg) ** 2)


def make_neighborhood_weights(
    cer_cx_rad,
    cer_cy_rad,
    cer_weights,
    pivot_radius_rad,
    min_num_neighborhood=2
):
    """
    Returns weights which depend less on individual outliers.
    """
    tree = HemisphereTree_init(cx=cer_cx_rad, cy=cer_cy_rad)
    return HemisphereTree_make_neighbor_weights(
        tree=tree,
        tree_weights=cer_weights,
        pivot_radius=pivot_radius_rad,
        min_num_neighborhood=min_num_neighborhood,
    )


def direction_cosines_to_vector(cx, cy):
    cz = np.sqrt(1.0 - cx ** 2 - cy ** 2)
    return np.c_[cx, cy, cz]


def HemisphereTree_init(cx, cy):
    vec = direction_cosines_to_vector(cx=cx, cy=cy)
    tree = scipy.spatial.cKDTree(data=vec)
    return tree

def HemisphereTree_query_self(tree, point_i, pivot_radius):
    """
    Returns distances 'dd' and indices 'ii' to neighboring points of 'point_i'.
    """
    _dd, _ii = tree.query(
        k=tree.data.shape[0],
        x=tree.data[point_i],
    )
    dd = []
    ii = []
    for i in range(len(_ii)):
        if _ii[i] == point_i:
            continue
        if _dd[i] > pivot_radius:
            continue
        dd.append(_dd[i])
        ii.append(_ii[i])
    return dd, ii


def HemisphereTree_make_neighbor_weights(
    tree,
    tree_weights,
    pivot_radius,
    min_num_neighborhood=2
):
    point_weights = []
    for point_i in range(tree.data.shape[0]):

        dd, ii = HemisphereTree_query_self(
            tree=tree,
            point_i=point_i,
            pivot_radius=pivot_radius
        )

        if len(ii) >= min_num_neighborhood:

            dw = (pivot_radius - dd) / pivot_radius
            dw[dw < 0] = 0

            w = tree_weights[ii]

            point_weight = np.average(w, weights=dw)
        else:
            point_weight = 0.0
        point_weights.append(point_weight)
    return np.array(point_weights)
