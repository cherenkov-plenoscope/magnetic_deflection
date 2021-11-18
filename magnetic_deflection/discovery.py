import numpy as np

from . import examples
from . import corsika
from . import light_field_characterization
from . import tools
from . import spherical_coordinates


def estimate_deflection(
    json_logger,
    prng,
    site,
    particle_energy,
    particle_id,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    density_cut,
    num_showers_per_iteration,
    max_num_showers,
    min_num_cherenkov_photons,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
):
    jlog = json_logger
    shift_angle_deg = 0.0
    prm_cone_deg = corsika.MAX_ZENITH_DEG
    prm_az_deg = 0.0
    prm_zd_deg = 0.0
    run_id = 0
    total_num_showers = 0
    last_off_axis_deg = 180.0
    cherenkov_pools = []

    guesses = []

    jlog.info("loop: start")
    jlog.info(
        "loop: {:d} shower/iteration, {:d} showers max.".format(
            num_showers_per_iteration, max_num_showers
        )
    )

    while True:
        run_id += 1

        if total_num_showers > max_num_showers:
            jlog.info("loop: break, too many showers")
            break

        total_num_showers += num_showers_per_iteration

        new_pools, _ = corsika.make_cherenkov_pools_statistics(
            site=site,
            particle_id=particle_id,
            particle_energy=particle_energy,
            particle_cone_azimuth_deg=prm_az_deg,
            particle_cone_zenith_deg=prm_zd_deg,
            particle_cone_opening_angle_deg=prm_cone_deg,
            num_showers=num_showers_per_iteration,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
            density_cut=density_cut,
            corsika_primary_path=corsika_primary_path,
            run_id=run_id,
            prng=prng,
        )

        min_num_valid_pools = int(np.ceil(0.1 * num_showers_per_iteration))
        num_new_pools = len(new_pools)
        cherenkov_pools += new_pools

        if num_new_pools < min_num_valid_pools:
            jlog.info(
                "loop: not enough valid cherenkov pools, "
                + "back to zenith, open cone, double num shower."
            )
            num_showers_per_iteration *= 2
            prm_cone_deg = corsika.MAX_ZENITH_DEG
            prm_az_deg = 0.0
            prm_zd_deg = 0.0
            continue

        off_axis_pivot_deg = (1 / 8) * (prm_cone_deg + max_off_axis_deg)

        guess = light_field_characterization.inspect_pools(
            cherenkov_pools=cherenkov_pools,
            off_axis_pivot_deg=off_axis_pivot_deg,
            instrument_azimuth_deg=instrument_azimuth_deg,
            instrument_zenith_deg=instrument_zenith_deg,
        )

        shift_angle_deg = spherical_coordinates._angle_between_az_zd_deg(
            az1_deg=prm_az_deg,
            zd1_deg=prm_zd_deg,
            az2_deg=guess["particle_azimuth_deg"],
            zd2_deg=guess["particle_zenith_deg"],
        )

        jlog.info(
            "loop: "
            + "azimuth {:.1f}, ".format(prm_az_deg)
            + "zenith {:.1f}, ".format(prm_zd_deg)
            + "opening {:.2f}, ".format(prm_cone_deg)
            + "off-axis {:.2f}, ".format(guess["off_axis_deg"])
            + "shift {:.2f} ".format(shift_angle_deg)
            + "all/deg, "
            + "num showers: {:d}".format(len(cherenkov_pools))
        )

        guesses.append(guess)

        if guess["off_axis_deg"] <= max_off_axis_deg:
            jlog.info("loop: return, off_axis_deg < max_off_axis_deg")
            return guesses

        prm_az_deg = float(guess["particle_azimuth_deg"])
        prm_zd_deg = float(guess["particle_zenith_deg"])

        if np.isnan(prm_az_deg) or np.isnan(prm_zd_deg):
            jlog.info("loop: break, particle directions are nan")
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
