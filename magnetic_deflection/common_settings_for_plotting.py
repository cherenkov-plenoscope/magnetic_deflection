import numpy as np
import spherical_coordinates


def hemisphere_field_of_view():
    TAU = 2 * np.pi
    out = {
        "wide": {
            "half_angle_rad": np.pi / 2,
            "zenith_mayor_rad": [],
            "zenith_minor_rad": np.deg2rad(
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            ),
            "azimuth_minor_rad": np.linspace(0, TAU, 36, endpoint=False),
        },
        "narrow": {
            "half_angle_rad": np.pi * 1 / 9,
            "zenith_mayor_rad": [],
            "zenith_minor_rad": np.deg2rad([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "azimuth_minor_rad": np.linspace(0, TAU, 36, endpoint=False),
        },
    }
    out["wide"]["rfov"] = np.sin(out["wide"]["half_angle_rad"])
    out["narrow"]["rfov"] = np.sin(out["narrow"]["half_angle_rad"])
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
    mag_az, mag_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
        cx=mag[0] * np.sign(mag[1]), cy=0.0, cz=np.abs(mag[1])
    )
    return {
        "azimuth_rad": mag_az,
        "zenith_rad": mag_zd,
        "magnitude_uT": magnitude_uT,
        "sign": np.sign(mag[1]),
    }


def make_great_circle_line(
    start_azimuth_rad,
    start_zenith_rad,
    stop_azimuth_rad,
    stop_zenith_rad,
    fN=100,
):
    """
    Draw a greact circle line between a start and a stop.
    The line is interpolated by fN points.

    Parameters
    ----------
    start_azimuth_rad : float
        Start point's azimuth
    start_zenith_rad : float
        Start point's zenith
    stop_azimuth_rad : float
        Stop point's azimuth
    stop_zenith_rad : float
        Stop point's zenith.
    fN : int (default 100)
        Number of points on the line

    Returns
    -------
    points on line : np.array(shape=(fN, 2))
        The azimuth_rad and zenith_rad of the points on the line.
    """
    start = np.array(
        spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=start_azimuth_rad,
            zenith_rad=start_zenith_rad,
        )
    )
    stop = np.array(
        spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=stop_azimuth_rad,
            zenith_rad=stop_zenith_rad,
        )
    )
    rot_axis = np.cross(start, stop)
    alpha_rad = spherical_coordinates.angle_between_az_zd(
        azimuth1_rad=start_azimuth_rad,
        zenith1_rad=start_zenith_rad,
        azimuth2_rad=stop_azimuth_rad,
        zenith2_rad=stop_zenith_rad,
    )

    points = np.zeros(shape=(fN, 3))
    alphas_rad = np.linspace(0.0, alpha_rad, fN)
    for i in range(fN):
        _t = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "rot": {
                "repr": "axis_angle",
                "axis": rot_axis,
                "angle_deg": np.rad2deg(alphas_rad[i]),
            },
        }
        t = homogeneous_transformation.compile(_t)
        points[i] = homogeneous_transformation.transform_orientation(
            t=t,
            d=start,
        )
    return spherical_coordinates.cx_cy_cz_to_az_zd(
        cx=points[:, 0], cy=points[:, 1], cz=points[:, 2]
    )
