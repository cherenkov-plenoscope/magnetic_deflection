import numpy as np
from . import discovery
import tempfile
import corsika_primary_wrapper as cpw
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


def make_corsika_primary_steering(
    run_id,
    site,
    num_events,
    primary_particle_id,
    primary_energy,
    primary_cx,
    primary_cy,
):
    steering = {}
    steering["run"] = {
        "run_id": int(run_id),
        "event_id_of_first_event": 1,
        "observation_level_asl_m": site["observation_level_asl_m"],
        "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
        "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
        "atmosphere_id": site["atmosphere_id"],
    }
    steering["primaries"] = []
    for event_id in range(num_events):
        az_deg, zd_deg = discovery._cx_cy_to_az_zd_deg(
            cx=primary_cx, cy=primary_cy
        )
        prm = {
            "particle_id": int(primary_particle_id),
            "energy_GeV": float(primary_energy),
            "zenith_rad": np.deg2rad(zd_deg),
            "azimuth_rad": np.deg2rad(az_deg),
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw.simple_seed(event_id + run_id * num_events),
        }
        steering["primaries"].append(prm)
    return steering

"""
def characterize_cherenkov_pool(
    site,
    primary_particle_id,
    primary_energy,
    primary_azimuth_deg,
    primary_zenith_deg,
    corsika_primary_path,
    total_energy_thrown=1e3,
    min_num_cherenkov_photons=1e2,
    outlier_percentile=100.0,
):
    assert total_energy_thrown > primary_energy
    num_airshower = int(np.ceil(total_energy_thrown / primary_energy))

    primary_cx, primary_cy = discovery._az_zd_to_cx_cy(
        azimuth_deg=primary_azimuth_deg, zenith_deg=primary_zenith_deg
    )

    corsika_primary_steering = make_corsika_primary_steering(
        run_id=1,
        site=site,
        num_events=num_airshower,
        primary_particle_id=primary_particle_id,
        primary_energy=primary_energy,
        primary_cx=primary_cx,
        primary_cy=primary_cy,
    )

    pools = []
    with tempfile.TemporaryDirectory(prefix="mag_defl_") as tmp:
        corsika_output_path = os.path.join(tmp, "run.tario")
        cpw.corsika_primary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_primary_steering,
            output_path=corsika_output_path,
        )
        run = cpw.Tario(corsika_output_path)
        for idx, airshower in enumerate(run):
            corsika_event_header, bunches = airshower
            num_bunches = bunches.shape[0]
            if num_bunches >= min_num_cherenkov_photons:
                pools.append(bunches)

        num_airshowers_found = len(pools)
        bunches = np.vstack(pools)
        inlier = mask_inlier_in_light_field_geometry(
            light_field=
            xs=bunches[:, cpw.IX] * cpw.CM2M,
            ys=bunches[:, cpw.IY] * cpw.CM2M,
            cxs=bunches[:, cpw.ICX],
            cys=bunches[:, cpw.ICY],
            outlier_percentile=outlier_percentile,
        )
        bunches_inlier = bunches[inlier]

        _out = parameterize_light_field(
            xs=bunches_inlier[:, cpw.IX] * cpw.CM2M,
            ys=bunches_inlier[:, cpw.IY] * cpw.CM2M,
            cxs=bunches_inlier[:, cpw.ICX],
            cys=bunches_inlier[:, cpw.ICY],
            ts=bunches_inlier[:, cpw.ITIME] * 1e-9,  # ns to s
        )
        _out["total_num_photons"] = float(np.sum(bunches_inlier[:, cpw.IBSIZE]))
        _out["total_num_airshowers"] = int(num_airshowers_found)
        _out["outlier_percentile"] = float(outlier_percentile)
        out = {}
        for key in _out:
            out[KEYPREFIX + key] = _out[key]
        return out
"""

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
    out["position_std_minor"] = xy_ellipse["minor_std"]
    out["position_std_major"] = xy_ellipse["major_std"]
    del xy_ellipse

    cxcy_ellipse = estimate_ellipse(a=lf["cx"], b=lf["cy"])
    out["direction_median_cx_rad"] = cxcy_ellipse["median_a"]
    out["direction_median_cy_rad"] = cxcy_ellipse["median_b"]
    out["direction_phi_rad"] = cxcy_ellipse["phi_rad"]
    out["direction_std_minor"] = cxcy_ellipse["minor_std"]
    out["direction_std_major"] = cxcy_ellipse["major_std"]
    del cxcy_ellipse

    out["arrival_time_mean_s"] = float(np.mean(lf["t"]))
    out["arrival_time_median_s"] = float(np.median(lf["t"]))
    out["arrival_time_std_s"] = float(np.std(lf["t"]))
    return out
