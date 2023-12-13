import os
import numpy as np
import binning_utils
from . import spherical_coordinates


def hemisphere_field_of_view():
    out = {
        "wide": {
            "half_angle_deg": 90,
            "zenith_mayor_deg": [],
            "zenith_minor_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "azimuth_minor_deg": np.linspace(0, 360, 36, endpoint=False),
        },
        "narrow": {
            "half_angle_deg": 10,
            "zenith_mayor_deg": [],
            "zenith_minor_deg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "azimuth_minor_deg": np.linspace(0, 360, 36, endpoint=False),
        },
    }
    out["wide"]["rfov"] = np.sin(np.deg2rad(out["wide"]["half_angle_deg"]))
    out["narrow"]["rfov"] = np.sin(np.deg2rad(out["narrow"]["half_angle_deg"]))
    return out


def common_energy_limits():
    return {"energy_start_GeV": 0.1, "energy_stop_GeV": 100}


def magnetic_flux(
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
):
    mag = np.array([earth_magnetic_field_x_muT, earth_magnetic_field_z_muT])
    magnitude_uT = np.linalg.norm(mag)
    mag = mag / magnitude_uT
    mag_az_deg, mag_zd_deg = spherical_coordinates._cx_cy_cz_to_az_zd_deg(
        cx=mag[0] * np.sign(mag[1]), cy=0.0, cz=np.abs(mag[1])
    )
    return {
        "azimuth_deg": mag_az_deg,
        "zenith_deg": mag_zd_deg,
        "magnitude_uT": magnitude_uT,
        "sign": np.sign(mag[1]),
    }
