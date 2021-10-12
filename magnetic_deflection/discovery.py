import json
import os
import numpy as np
import corsika_primary_wrapper as cpw
import pandas
import tempfile
import time
from . import examples
from . import corsika


def _azimuth_range(azimuth_deg):
    # Enforce azimuth between -180deg and +180deg
    azimuth_deg = azimuth_deg % 360
    # force it to be the positive remainder, so that 0 <= angle < 360
    azimuth_deg = (azimuth_deg + 360) % 360
    # force into the minimum absolute value residue class,
    # so that -180 < angle <= 180
    if np.isscalar(azimuth_deg):
        if azimuth_deg > 180:
            azimuth_deg -= 360
    else:
        azimuth_deg = np.asarray(azimuth_deg)
        mask = azimuth_deg > 180.0
        azimuth_deg[mask] -= 360.0
    return azimuth_deg


def _az_zd_to_cx_cy(azimuth_deg, zenith_deg):
    azimuth_deg = _azimuth_range(azimuth_deg)
    # Adopted from CORSIKA
    az = np.deg2rad(azimuth_deg)
    zd = np.deg2rad(zenith_deg)
    cx = np.cos(az) * np.sin(zd)
    cy = np.sin(az) * np.sin(zd)
    _cz = np.cos(zd)
    return cx, cy


def _cx_cy_to_az_zd_deg(cx, cy):
    inner_sqrt = 1.0 - cx ** 2 - cy ** 2
    if np.isscalar(inner_sqrt):
        if inner_sqrt >= 0:
            cz = np.sqrt(inner_sqrt)
        else:
            cz = float("nan")
    else:
        cz = np.nan * np.ones(len(inner_sqrt))
        fine = inner_sqrt >= 0
        cz[fine] = np.sqrt(inner_sqrt)
    # 1 = sqrt(cx**2 + cy**2 + cz**2)
    az = np.arctan2(cy, cx)
    zd = np.arccos(cz)
    return np.rad2deg(az), np.rad2deg(zd)


def _angle_between_az_zd_deg(az1_deg, zd1_deg, az2_deg, zd2_deg):
    az1 = np.deg2rad(az1_deg)
    zd1 = np.deg2rad(zd1_deg)
    az2 = np.deg2rad(az2_deg)
    zd2 = np.deg2rad(zd2_deg)
    return np.rad2deg(
        _great_circle_distance_long_lat(
            lam_long1=az1,
            phi_alt1=np.pi / 2 - zd1,
            lam_long2=az2,
            phi_alt2=np.pi / 2 - zd2,
        )
    )


def _great_circle_distance_long_lat(lam_long1, phi_alt1, lam_long2, phi_alt2):
    delta_lam = np.abs(lam_long2 - lam_long1)
    delta_sigma = np.arccos(
        np.sin(phi_alt1) * np.sin(phi_alt2)
        + np.cos(phi_alt1) * np.cos(phi_alt2) * np.cos(delta_lam)
    )
    return delta_sigma


def direct_discovery(
    run_id,
    num_events,
    primary_particle_id,
    primary_energy,
    best_primary_azimuth_deg,
    best_primary_zenith_deg,
    spray_radius_deg,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    site,
    prng,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
    min_num_cherenkov_photons_in_airshower=100,
    DEBUG_PRINT=True,
):
    out = {
        "iteration": int(run_id),
        "primary_azimuth_deg": float("nan"),
        "primary_zenith_deg": float("nan"),
        "primary_cone_opening_angle_deg": float(spray_radius_deg),
        "off_axis_deg": float("nan"),
        "cherenkov_pool_x_m": float("nan"),
        "cherenkov_pool_y_m": float("nan"),
        "cherenkov_pool_cx": float("nan"),
        "cherenkov_pool_cy": float("nan"),
        "num_valid_Cherenkov_pools": 0,
        "num_thrown_Cherenkov_pools": int(num_events),
        "valid": False,
    }

    steering = corsika.make_steering(
        run_id=run_id,
        site=site,
        primary_particle_id=primary_particle_id,
        primary_energy=primary_energy,
        primary_cone_azimuth_deg=best_primary_azimuth_deg,
        primary_cone_zenith_deg=best_primary_zenith_deg,
        primary_cone_opening_angle_deg=spray_radius_deg,
        num_events=num_events,
        prng=prng,
    )

    cherenkov_pools = corsika.estimate_cherenkov_pool(
        corsika_primary_steering=steering,
        corsika_primary_path=corsika_primary_path,
        min_num_cherenkov_photons=min_num_cherenkov_photons_in_airshower,
        outlier_percentile=100.0,
    )
    cherenkov_pools = pandas.DataFrame(cherenkov_pools)
    cherenkov_pools = cherenkov_pools.to_records(index=False)

    actual_num_valid_pools = len(cherenkov_pools)
    expected_num_valid_pools = int(np.ceil(0.1 * num_events))
    if actual_num_valid_pools < expected_num_valid_pools:
        out["valid"] = False
        return out

    (
        cherenkov_pool_azimuth_deg,
        cherenkov_pool_zenith_deg
    ) = _cx_cy_to_az_zd_deg(
        cx=cherenkov_pools["direction_median_cx_rad"],
        cy=cherenkov_pools["direction_median_cy_rad"],
    )

    delta_c_deg = _angle_between_az_zd_deg(
        az1_deg=cherenkov_pool_azimuth_deg,
        zd1_deg=cherenkov_pool_zenith_deg,
        az2_deg=instrument_azimuth_deg,
        zd2_deg=instrument_zenith_deg,
    )

    c_ref_deg = (1/8) * (spray_radius_deg + max_off_axis_deg)
    weights = np.exp(-0.5 * (delta_c_deg / c_ref_deg ) ** 2)

    prm_az = np.average(
        cherenkov_pools["primary_azimuth_rad"], weights=weights
    )
    prm_zd = np.average(
        cherenkov_pools["primary_zenith_rad"], weights=weights
    )
    average_off_axis_deg = np.average(delta_c_deg, weights=weights)

    out["valid"] = True
    out["primary_azimuth_deg"] = float(np.rad2deg(prm_az))
    out["primary_zenith_deg"] = float(np.rad2deg(prm_zd))
    out["off_axis_deg"] = float(average_off_axis_deg)

    out["cherenkov_pool_x_m"] = float(
        np.average(cherenkov_pools["position_median_x_m"], weights=weights)
    )
    out["cherenkov_pool_y_m"] = float(
        np.average(cherenkov_pools["position_median_y_m"], weights=weights)
    )
    out["cherenkov_pool_cx"] = float(
        np.average(cherenkov_pools["direction_median_cx_rad"], weights=weights)
    )
    out["cherenkov_pool_cy"] = float(
        np.average(cherenkov_pools["direction_median_cy_rad"], weights=weights)
    )
    _prm_cx, _prm_cy = _az_zd_to_cx_cy(
        azimuth_deg=out["primary_azimuth_deg"],
        zenith_deg=out["primary_zenith_deg"],
    )
    out["primary_cx"] = float(_prm_cx)
    out["primary_cy"] = float(_prm_cy)

    out["num_valid_Cherenkov_pools"] = len(cherenkov_pools)
    out["num_thrown_Cherenkov_pools"] = int(num_events)

    if DEBUG_PRINT:
        print(json.dumps(out, indent=4))
        asw = np.argsort(weights)
        for i in range(int(len(asw) / 10)):
            j = asw[len(asw) - 1 - i]
            print("weight {:03d}%, delta_c {:.2f}deg".format(
                    int(100*weights[j]),
                    delta_c_deg[j],
                )
            )

    return out


def estimate_deflection(
    prng,
    site,
    primary_energy,
    primary_particle_id,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    initial_num_events_per_iteration=2 ** 5,
    max_total_num_events=2 ** 13,
    min_num_valid_Cherenkov_pools=100,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
    iteration_speed=0.9,
    min_num_cherenkov_photons_in_airshower=100,
    verbose=True,
):
    spray_radius_deg = cpw.MAX_ZENITH_DEG
    prm_az_deg = 0.0
    prm_zd_deg = 0.0
    run_id = 0
    total_num_events = 0
    num_events = initial_num_events_per_iteration * 8
    while True:
        run_id += 1

        total_num_events += num_events
        guess = direct_discovery(
            run_id=run_id,
            num_events=num_events,
            primary_particle_id=primary_particle_id,
            primary_energy=primary_energy,
            best_primary_azimuth_deg=prm_az_deg,
            best_primary_zenith_deg=prm_zd_deg,
            spray_radius_deg=spray_radius_deg,
            instrument_azimuth_deg=instrument_azimuth_deg,
            instrument_zenith_deg=instrument_zenith_deg,
            max_off_axis_deg=max_off_axis_deg,
            site=site,
            prng=prng,
            corsika_primary_path=corsika_primary_path,
            min_num_cherenkov_photons_in_airshower=(
                min_num_cherenkov_photons_in_airshower
            ),
        )

        if (
            guess["valid"]
            and guess["off_axis_deg"] <= max_off_axis_deg
        ):
            guess["total_num_events"] = total_num_events
            return guess

        if spray_radius_deg < max_off_axis_deg:
            print("spray_radius_deg < max_off_axis_deg")
            break

        if spray_radius_deg < guess["off_axis_deg"]:
            num_events *= 2
            spray_radius_deg *= np.sqrt(2.0)
            print("double num events.")
            continue

        if total_num_events > max_total_num_events:
            print("Too many events thrown.")
            break

        spray_radius_deg *= 1.0 / np.sqrt(2.0)
        prm_az_deg = guess["primary_azimuth_deg"]
        prm_zd_deg = guess["primary_zenith_deg"]

        if np.isnan(prm_az_deg) or np.isnan(prm_zd_deg):
            print("directions are Nan")
            break

    # failed.
    return {"valid": False}
