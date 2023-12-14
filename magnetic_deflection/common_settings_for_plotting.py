import numpy as np
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


def make_great_circle_line(
    start_azimuth_deg,
    start_zenith_deg,
    stop_azimuth_deg,
    stop_zenith_deg,
    fN=100,
):
    """
    Draw a greact circle line between a start and a stop.
    The line is interpolated by fN points.

    Parameters
    ----------
    start_azimuth_deg : float
        Start point's azimuth
    start_zenith_deg : float
        Start point's zenith
    stop_azimuth_deg : float
        Stop point's azimuth
    stop_zenith_deg : float
        Stop point's zenith.
    fN : int (default 100)
        Number of points on the line

    Returns
    -------
    points on line : np.array(shape=(fN, 2))
        The azimuth_deg and zenith_deg of the points on the line.
    """
    start = np.array(spherical_coordinates._az_zd_to_cx_cy_cz(start_azimuth_deg, start_zenith_deg))
    stop = np.array(spherical_coordinates._az_zd_to_cx_cy_cz(stop_azimuth_deg, stop_zenith_deg))
    rot_axis = np.cross(start, stop)
    alpha = spherical_coordinates._angle_between_vectors_rad(start, stop)

    points = np.zeros(shape=(fN, 3))
    alphas_deg = np.rad2deg(np.linspace(0.0, alpha, fN))
    for i in range(fN):
        _t = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "rot": {
                "repr": "axis_angle",
                "axis": rot_axis,
                "angle_deg": alphas_deg[i],
            },
        }
        t = homogeneous_transformation.compile(_t)
        points[i] = homogeneous_transformation.transform_orientation(
            t=t,
            d=start,
        )
    return spherical_coordinates._cx_cy_cz_to_az_zd_deg(
        cx=points[:, 0], cy=points[:, 1], cz=points[:, 2]
    )
