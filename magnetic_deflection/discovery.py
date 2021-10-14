import json
import os
import numpy as np
import corsika_primary_wrapper as cpw
import pandas
import tempfile
import time
from . import examples
from . import corsika
from . import spherical_coordinates as sphcords
from . import light_field_characterization
from . import tools


def direct_discovery(
    cherenkov_pools,
    run_id,
    num_showers,
    particle_id,
    particle_energy,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    site,
    prng,
    outlier_percentile,
    min_num_cherenkov_photons,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
):
    out = {}

    steering = corsika.make_steering(
        run_id=run_id,
        site=site,
        particle_id=particle_id,
        particle_energy=particle_energy,
        particle_cone_azimuth_deg=particle_cone_azimuth_deg,
        particle_cone_zenith_deg=particle_cone_zenith_deg,
        particle_cone_opening_angle_deg=particle_cone_opening_angle_deg,
        num_showers=num_showers,
        prng=prng,
    )

    new_pools = corsika.estimate_cherenkov_pool(
        corsika_primary_steering=steering,
        corsika_primary_path=corsika_primary_path,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
        outlier_percentile=outlier_percentile,
    )

    expected_num_valid_pools = int(np.ceil(0.1 * num_showers))
    if len(new_pools) < expected_num_valid_pools:
        out["valid"] = False
        return out

    cherenkov_pools += new_pools

    off_axis_pivot_deg = (1 / 8) * (
        particle_cone_opening_angle_deg + max_off_axis_deg
    )
    insp = light_field_characterization.inspect_pools(
        cherenkov_pools=cherenkov_pools,
        off_axis_pivot_deg=off_axis_pivot_deg,
        instrument_azimuth_deg=instrument_azimuth_deg,
        instrument_zenith_deg=instrument_zenith_deg,
        debug_print=False,
    )

    out["valid"] = True
    out.update(insp)
    return out


def estimate_deflection(
    json_logger,
    prng,
    site,
    particle_energy,
    particle_id,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    outlier_percentile,
    num_showers_per_iteration,
    max_num_showers,
    min_num_cherenkov_photons,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
    guesses_path=None,
):
    jlog = json_logger
    prm_cone_deg = cpw.MAX_ZENITH_DEG
    prm_az_deg = 0.0
    prm_zd_deg = 0.0
    run_id = 0
    total_num_showers = 0
    last_off_axis_deg = 180.0
    cherenkov_pools = []

    guesses = []

    jlog.info("loop: start")
    jlog.info("loop: {:d} shower/iteration".format(num_showers_per_iteration))

    while True:
        run_id += 1

        total_num_showers += num_showers_per_iteration
        guess = direct_discovery(
            cherenkov_pools=cherenkov_pools,
            run_id=run_id,
            num_showers=num_showers_per_iteration,
            particle_id=particle_id,
            particle_energy=particle_energy,
            particle_cone_azimuth_deg=prm_az_deg,
            particle_cone_zenith_deg=prm_zd_deg,
            particle_cone_opening_angle_deg=prm_cone_deg,
            instrument_azimuth_deg=instrument_azimuth_deg,
            instrument_zenith_deg=instrument_zenith_deg,
            max_off_axis_deg=max_off_axis_deg,
            site=site,
            prng=prng,
            corsika_primary_path=corsika_primary_path,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
            outlier_percentile=outlier_percentile,
        )

        jlog.info(
            "loop: azimuth {:.1f}, zenith {:.1f}, opening {:.2f}, off-axis {:.2f} all/deg, num showers: {:d}".format(
                prm_az_deg,
                prm_zd_deg,
                prm_cone_deg,
                guess["off_axis_deg"],
                len(cherenkov_pools),
            )
        )

        guesses.append(guess)
        if guesses_path:
            tools.append_jsonl_unsave(guesses_path, guess)

        if guess["valid"] and guess["off_axis_deg"] <= max_off_axis_deg:
            guess["total_num_showers"] = total_num_showers
            jlog.info("loop: return, off_axis_deg < max_off_axis_deg")
            return guesses

        prm_az_deg = float(guess["particle_azimuth_deg"])
        prm_zd_deg = float(guess["particle_zenith_deg"])

        if np.isnan(prm_az_deg) or np.isnan(prm_zd_deg):
            jlog.info("loop: break, particle directions are nan")
            break

        if total_num_showers > max_num_showers:
            jlog.info("loop: break, too many showers")
            break

        if prm_cone_deg < max_off_axis_deg:
            prm_cone_deg *= 2.0
            jlog.info(
                "loop: increase opening to {:.1f}deg".format(prm_cone_deg)
            )
        else:
            prm_cone_deg *= 1.0 / np.sqrt(2.0)

        if (
            guess["off_axis_deg"] > last_off_axis_deg
            and guess["off_axis_deg"] > (1 / 2) * prm_cone_deg
        ):
            num_showers_per_iteration *= 2
            prm_cone_deg *= 2.0
            jlog.info(
                "loop: increase num showers to {:d}, and keep opening".format(
                    num_showers_per_iteration
                )
            )

        last_off_axis_deg = float(guess["off_axis_deg"])

    jlog.info("loop: failed")
    return guesses
