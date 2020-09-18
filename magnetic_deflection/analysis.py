import numpy as np
import pandas as pd
import scipy


"""
prepare deflection_table
========================
"""


def add_density_fields_to_deflection_table(deflection_table):
    out = {}
    for site_key in deflection_table:
        out[site_key] = {}
        for particle_key in deflection_table[site_key]:
            t = deflection_table[site_key][particle_key]
            dicout = pd.DataFrame(t).to_dict(orient="list")

            dicout['num_cherenkov_photons_per_shower'] = (
                t['char_total_num_photons'] /
                t['char_total_num_airshowers'])

            dicout['spread_area_m2'] = (
                np.pi *
                t['char_position_std_major_m'] *
                t['char_position_std_minor_m'])

            dicout['spread_solid_angle_deg2'] = (
                np.pi *
                np.rad2deg(t['char_direction_std_major_rad']) *
                np.rad2deg(t['char_direction_std_minor_rad']))

            dicout['light_field_outer_density'] = (
                dicout['num_cherenkov_photons_per_shower'] /
                (dicout['spread_solid_angle_deg2']*dicout['spread_area_m2']))
            out[site_key][particle_key] = pd.DataFrame(dicout).to_records(
                index=False)
    return out


def cut_invalid_from_deflection_table(
    deflection_table,
    but_keep_site="Off",
    min_energy=1e-1,
):
    out = {}
    for site_key in deflection_table:
        if but_keep_site in site_key:
            out[site_key] = deflection_table[site_key]
        else:
            out[site_key] = {}
            for particle_key in deflection_table[site_key]:
                t_raw = deflection_table[site_key][particle_key]
                defelction_valid = t_raw['primary_azimuth_deg'] != 0.
                energy_valid = t_raw['energy_GeV'] >= min_energy
                valid = np.logical_and(energy_valid, defelction_valid)
                out[site_key][particle_key] = t_raw[valid]
    return out


"""
Reject outliers, smooth
=======================
"""


def percentile_indices(values, target_value, percentile=90):
    values = np.array(values)
    factor = percentile/100.
    delta = np.abs(values - target_value)
    argsort_delta = np.argsort(delta)
    num_values = len(values)
    idxs = np.arange(num_values)
    idxs_sorted = idxs[argsort_delta]
    idx_limit = int(np.ceil(num_values*factor))
    return idxs_sorted[0: idx_limit]


def smooth(energies, values):
    suggested_num_energy_bins = int(np.ceil(2*np.sqrt(len(values))))
    suggested_energy_bin_edges = np.geomspace(
        np.min(energies),
        np.max(energies),
        suggested_num_energy_bins+1)
    suggested_energy_supports = 0.5*(
        suggested_energy_bin_edges[0:-1] +
        suggested_energy_bin_edges[1:])

    actual_energy_supports = []
    key_med = []
    key_mean80 = []
    key_std80 = []
    for ibin in range(len(suggested_energy_bin_edges) - 1):
        e_start = suggested_energy_bin_edges[ibin]
        e_stop = suggested_energy_bin_edges[ibin+1]
        mask = np.logical_and(energies >= e_start, energies < e_stop)
        if np.sum(mask) > 3:
            actual_energy_supports.append(suggested_energy_supports[ibin])
            med = np.median(values[mask])
            key_med.append(med)
            indices80 = percentile_indices(
                values=values[mask],
                target_value=med,
                percentile=80)
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
    return scale*energy**(index)
