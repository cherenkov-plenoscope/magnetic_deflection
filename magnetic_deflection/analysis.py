import numpy as np
import pandas
from . import corsika

from scipy.optimize import curve_fit as scipy_optimize_curve_fit


"""
prepare deflection_table
========================
"""


def add_density_fields_to_deflection(deflection):
    t = deflection
    dicout = pandas.DataFrame(t).to_dict(orient="list")

    dicout["num_cherenkov_photons_per_shower"] = (
        t["total_num_photons"] / t["total_num_showers"]
    )

    dicout["spread_area_m2"] = (
        np.pi * t["position_std_major_m"] * t["position_std_minor_m"]
    )

    dicout["spread_solid_angle_deg2"] = (
        np.pi
        * np.rad2deg(t["direction_std_major_rad"])
        * np.rad2deg(t["direction_std_minor_rad"])
    )

    dicout["light_field_outer_density"] = dicout[
        "num_cherenkov_photons_per_shower"
    ] / (dicout["spread_solid_angle_deg2"] * dicout["spread_area_m2"])
    return pandas.DataFrame(dicout).to_records(index=False)


def cut_invalid_from_deflection(deflection, min_energy):
    mask_az = deflection["particle_azimuth_deg"] != 0.0
    mask_en = deflection["particle_energy_GeV"] >= min_energy
    valid = np.logical_and(mask_en, mask_az)
    return deflection[valid]


FIT_KEYS = {
    "particle_azimuth_deg": {"start": 90.0,},
    "particle_zenith_deg": {"start": 0.0,},
    "position_med_x_m": {"start": 0.0,},
    "position_med_y_m": {"start": 0.0,},
}


def smooth_deflection_and_reject_outliers(deflection):
    sm = {}
    for key in FIT_KEYS:
        sres = smooth(
            energies=deflection["particle_energy_GeV"], values=deflection[key]
        )
        if "particle_energy_GeV" in sm:
            np.testing.assert_array_almost_equal(
                x=sm["particle_energy_GeV"],
                y=sres["energy_supports"],
                decimal=3,
            )
        else:
            sm["particle_energy_GeV"] = sres["energy_supports"]
        sm[key] = sres["key_mean80"]
        sm[key + "_std"] = sres["key_std80"]
    return pandas.DataFrame(sm)


def add_high_energy_to_deflection(
    deflection, charge_sign, energy_start=200, energy_stop=600, num_points=20,
):
    sm = {}
    for key in FIT_KEYS:
        sm["particle_energy_GeV"] = np.array(
            deflection["particle_energy_GeV"].tolist()
            + np.geomspace(energy_start, energy_stop, num_points).tolist()
        )
        key_start = charge_sign * FIT_KEYS[key]["start"]
        sm[key] = np.array(
            deflection[key].tolist()
            + (key_start * np.ones(num_points)).tolist()
        )
    df = pandas.DataFrame(sm)
    df = df[df["particle_zenith_deg"] <= corsika.MAX_ZENITH_DEG]
    return df.to_records(index=False)


def fit_power_law_to_deflection(
    deflection, charge_sign,
):
    t = deflection
    fits = {}
    for key in FIT_KEYS:
        key_start = charge_sign * FIT_KEYS[key]["start"]
        if np.mean(t[key] - key_start) > 0:
            sig = -1
        else:
            sig = 1

        try:
            expy, _ = scipy_optimize_curve_fit(
                power_law,
                t["particle_energy_GeV"],
                t[key] - key_start,
                p0=(sig * charge_sign, 1.0),
            )
        except RuntimeError as err:
            print(err)
            expy = [0.0, 0.0]

        fits[key] = {
            "formula": "f(energy) = scale*energy**index + offset",
            "scale": float(expy[0]),
            "index": float(expy[1]),
            "offset": float(key_start),
        }
    return fits


def make_fit_deflection(power_law_fit, particle, num_supports=1024):
    out = {}
    out["particle_energy_GeV"] = np.geomspace(
        np.min(particle["energy_bin_edges_GeV"]),
        np.max(particle["energy_bin_edges_GeV"]),
        num_supports,
    )
    for key in FIT_KEYS:
        rec_key = power_law(
            energy=out["particle_energy_GeV"],
            scale=power_law_fit[key]["scale"],
            index=power_law_fit[key]["index"],
        )
        rec_key += power_law_fit[key]["offset"]
        out[key] = rec_key

    df = pandas.DataFrame(out)
    df = df[df["particle_zenith_deg"] <= corsika.MAX_ZENITH_DEG]
    return df.to_records(index=False)


"""
Reject outliers, smooth
=======================
"""


def percentile_indices(values, target_value, percentile=90):
    values = np.array(values)
    factor = percentile / 100.0
    delta = np.abs(values - target_value)
    argsort_delta = np.argsort(delta)
    num_values = len(values)
    idxs = np.arange(num_values)
    idxs_sorted = idxs[argsort_delta]
    idx_limit = int(np.ceil(num_values * factor))
    return idxs_sorted[0:idx_limit]


def smooth(energies, values):
    suggested_num_energy_bins = int(np.ceil(2 * np.sqrt(len(values))))
    suggested_energy_bin_edges = np.geomspace(
        np.min(energies), np.max(energies), suggested_num_energy_bins + 1
    )
    suggested_energy_supports = 0.5 * (
        suggested_energy_bin_edges[0:-1] + suggested_energy_bin_edges[1:]
    )

    actual_energy_supports = []
    key_med = []
    key_mean80 = []
    key_std80 = []
    for ibin in range(len(suggested_energy_bin_edges) - 1):
        e_start = suggested_energy_bin_edges[ibin]
        e_stop = suggested_energy_bin_edges[ibin + 1]
        mask = np.logical_and(energies >= e_start, energies < e_stop)
        if np.sum(mask) > 3:
            actual_energy_supports.append(suggested_energy_supports[ibin])
            med = np.median(values[mask])
            key_med.append(med)
            indices80 = percentile_indices(
                values=values[mask], target_value=med, percentile=80
            )
            key_std80.append(np.std(values[mask][indices80]))
            key_mean80.append(np.mean(values[mask][indices80]))
    return {
        "energy_supports": np.array(actual_energy_supports),
        "key_med": np.array(key_med),
        "key_std80": np.array(key_std80),
        "key_mean80": np.array(key_mean80),
    }


"""
Fitting power-laws
==================
"""


def power_law(energy, scale, index):
    return scale * energy ** (index)
