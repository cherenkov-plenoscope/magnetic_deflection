import solid_angle_utils
import numpy as np


def query_cherenkov_ball_in_all_energy(
    allsky,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    min_num_cherenkov_photons=1e3,
):
    energy_GeV = allsky.config["binning"]["energy"]["stop_GeV"]
    energy_factor = 1.0 - (
        allsky.config["binning"]["energy"]["start_GeV"] / energy_GeV
    )

    return allsky.query_cherenkov_ball(
        azimuth_deg=azimuth_deg,
        zenith_deg=zenith_deg,
        energy_GeV=energy_GeV,
        half_angle_deg=half_angle_deg,
        energy_factor=energy_factor,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
    )


def estimate_cherenkov_density(showers, percentile=50):
    assert percentile in [50, 90]

    cer_solid_angle = "cherenkov_solid_angle{:d}_sr".format(percentile)
    cer_radius = "cherenkov_radius{:d}_m".format(percentile)
    cer_area = "cherenkov_area{:d}_m2".format(percentile)
    cer_half_angle = "cherenkov_half_angle{:d}_rad".format(percentile)
    cer_lf_density = "cherenkov_light_field_density{:d}_per_m2_per_sr".format(
        percentile
    )
    cer_sa_density = "cherenkov_solid_angle_density{:d}_per_sr".format(
        percentile
    )
    cer_a_density = "cherenkov_area_density{:d}_per_m2".format(percentile)

    factor = percentile * 1e-2

    out = {}
    out[cer_solid_angle] = solid_angle_utils.cone.solid_angle(
        half_angle_rad=showers[cer_half_angle]
    )
    out[cer_area] = np.pi * showers[cer_radius] ** 2.0

    out[cer_a_density] = (factor * showers["cherenkov_num_photons"]) / out[
        cer_area
    ]

    out[cer_sa_density] = (factor * showers["cherenkov_num_photons"]) / out[
        cer_solid_angle
    ]

    out[cer_lf_density] = (
        factor
        * showers["cherenkov_num_photons"]
        / (out[cer_solid_angle] * out[cer_area])
    )
    return out
