import json
import os
import numpy as np

from . import examples
from . import corsika
from . import light_field_characterization
from . import tools


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
    prm_cone_deg = corsika.MAX_ZENITH_DEG
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

        new_pools = corsika.make_cherenkov_pools_statistics(
            site=site,
            particle_id=particle_id,
            particle_energy=particle_energy,
            particle_cone_azimuth_deg=prm_az_deg,
            particle_cone_zenith_deg=prm_zd_deg,
            particle_cone_opening_angle_deg=prm_cone_deg,
            num_showers=num_showers_per_iteration,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
            outlier_percentile=outlier_percentile,
            corsika_primary_path=corsika_primary_path,
            run_id=run_id,
            prng=prng,
        )

        min_num_valid_pools = int(np.ceil(0.1 * num_showers_per_iteration))
        if len(new_pools) < min_num_valid_pools:
            jlog.info("loop: break, not enough valid cherenkov pools")
            break

        cherenkov_pools += new_pools

        off_axis_pivot_deg = (1 / 8) * (prm_cone_deg + max_off_axis_deg)

        guess = light_field_characterization.inspect_pools(
            cherenkov_pools=cherenkov_pools,
            off_axis_pivot_deg=off_axis_pivot_deg,
            instrument_azimuth_deg=instrument_azimuth_deg,
            instrument_zenith_deg=instrument_zenith_deg,
            debug_print=False,
        )

        jlog.info(
            "loop: "
            + "azimuth {:.1f}, ".format(prm_az_deg)
            + "zenith {:.1f}, ".format(prm_zd_deg)
            + "opening {:.2f}, ".format(prm_cone_deg)
            + "off-axis {:.2f} ".format(guess["off_axis_deg"])
            + "all/deg, "
            + "num showers: {:d}".format(len(cherenkov_pools))
        )

        guesses.append(guess)
        if guesses_path:
            tools.append_jsonl_unsave(guesses_path, guess)

        if guess["off_axis_deg"] <= max_off_axis_deg:
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

        if prm_cone_deg < 2 * guess["off_axis_deg"]:
            prm_cone_deg *= 2.0
            prm_cone_deg = np.min([prm_cone_deg, corsika.MAX_ZENITH_DEG])
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
