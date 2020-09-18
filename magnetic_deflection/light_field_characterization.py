import numpy as np
from . import discovery
import tempfile
import corsika_primary_wrapper as cpw
import os
import pandas as pd


KEYPREFIX = 'char_'

KEYS = [
    'char_position_med_x_m',
    'char_position_med_y_m',
    'char_position_phi_rad',
    'char_position_std_major_m',
    'char_position_std_minor_m',
    'char_direction_med_cx_rad',
    'char_direction_med_cy_rad',
    'char_direction_phi_rad',
    'char_direction_std_major_rad',
    'char_direction_std_minor_rad',
    'char_arrival_time_mean_s',
    'char_arrival_time_median_s',
    'char_arrival_time_std_s',
    'char_total_num_photons',
    'char_total_num_airshowers',
    'char_outlier_percentile',
]


def characterize_cherenkov_pool(
    site,
    primary_particle_id,
    primary_energy,
    primary_azimuth_deg,
    primary_zenith_deg,
    corsika_primary_path,
    total_energy_thrown=1e3,
    min_num_cherenkov_photons=1e2,
    outlier_percentile=100.,
):
    assert total_energy_thrown > primary_energy
    num_airshower = int(np.ceil(total_energy_thrown/primary_energy))

    primary_cx, primary_cy = discovery._az_zd_to_cx_cy(
        azimuth_deg=primary_azimuth_deg,
        zenith_deg=primary_zenith_deg)

    corsika_primary_steering = discovery._make_steering(
        run_id=1,
        site=site,
        num_events=num_airshower,
        primary_particle_id=primary_particle_id,
        primary_energy=primary_energy,
        primary_cx=primary_cx,
        primary_cy=primary_cy)

    pools = []
    with tempfile.TemporaryDirectory(prefix="mag_defl_") as tmp:
        corsika_output_path = os.path.join(tmp, "run.tario")
        cpw.corsika_primary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_primary_steering,
            output_path=corsika_output_path)
        run = cpw.Tario(corsika_output_path)
        for idx, airshower in enumerate(run):
            corsika_event_header, bunches = airshower
            num_bunches = bunches.shape[0]
            if num_bunches >= min_num_cherenkov_photons:
                pools.append(bunches)

        num_airshowers_found = len(pools)
        bunches = np.vstack(pools)
        _out = parameterize_light_field(
            xs=bunches[:, cpw.IX]*cpw.CM2M,
            ys=bunches[:, cpw.IY]*cpw.CM2M,
            cxs=bunches[:, cpw.ICX],
            cys=bunches[:, cpw.ICY],
            ts=bunches[:, cpw.ITIME]*1e-9,  # ns to s
            outlier_percentile=outlier_percentile)
        _out["total_num_photons"] = float(np.sum(bunches[:, cpw.IBSIZE]))
        _out['total_num_airshowers'] = int(num_airshowers_found)
        _out['outlier_percentile'] = float(outlier_percentile)
        out = {}
        for key in _out:
            out[KEYPREFIX+key] = _out[key]
        return out


def estimate_ellipse(cxs, cys, prefix="", direction_c="", unit=""):
    cov_matrix = np.cov(np.c_[cxs, cys].T)
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

    dc = direction_c
    return {
        prefix+"med_"+dc+"x"+unit: float(np.median(cxs)),
        prefix+"med_"+dc+"y"+unit: float(np.median(cys)),
        prefix+"phi_rad": float(phi),
        prefix+"std_major"+unit: float(major_std),
        prefix+"std_minor"+unit: float(minor_std)
    }


def percentile_indices(values, target_value, percentile):
    """
    Rejecting outliers.
    Return a mask indicating the percentile of elements which are
    closest to the target_value.
    """
    factor = percentile/100.
    delta = np.abs(values - target_value)
    argsort_delta = np.argsort(delta)
    num_values = len(values)
    idxs = np.arange(num_values)
    idxs_sorted = idxs[argsort_delta]
    idx_limit = int(np.ceil(num_values*factor))
    valid_indices = idxs_sorted[0: idx_limit]
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
        values=values,
        target_value=np.median(values),
        percentile=percentile)


def find_outlier_in_light_field_geometry(xs, ys, cxs, cys, percentile):
    valid_x = percentile_indices_wrt_median(
        values=xs,
        percentile=percentile)
    valid_y = percentile_indices_wrt_median(
        values=ys,
        percentile=percentile)
    valid_cx = percentile_indices_wrt_median(
        values=cxs,
        percentile=percentile)
    valid_cy = percentile_indices_wrt_median(
        values=cys,
        percentile=percentile)
    return np.logical_and(
        np.logical_and(valid_cx, valid_cy),
        np.logical_and(valid_x, valid_y))


def parameterize_light_field(xs, ys, cxs, cys, ts, outlier_percentile=100.0):
    if outlier_percentile != 100.0:
        valid = find_outlier_in_light_field_geometry(
            xs=xs,
            ys=ys,
            cxs=cxs,
            cys=cys,
            percentile=outlier_percentile)
        xs = xs[valid]
        ys = ys[valid]
        cxs = cxs[valid]
        cys = cys[valid]
        ts = ts[valid]
    xy_ellipse = estimate_ellipse(
        xs,
        ys,
        prefix="position_",
        unit='_m')
    cxcy_ellipse = estimate_ellipse(
        cxs,
        cys,
        prefix="direction_",
        unit='_rad',
        direction_c='c')
    out = {}
    out.update(xy_ellipse)
    out.update(cxcy_ellipse)
    out["arrival_time_mean_s"] = float(np.mean(ts))
    out["arrival_time_median_s"] = float(np.median(ts))
    out["arrival_time_std_s"] = float(np.std(ts))
    return out
