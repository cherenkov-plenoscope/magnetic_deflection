import numpy as np


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


def _az_zd_to_cx_cy_cz(azimuth_deg, zenith_deg):
    azimuth_deg = _azimuth_range(azimuth_deg)
    # Adopted from CORSIKA
    az = np.deg2rad(azimuth_deg)
    zd = np.deg2rad(zenith_deg)
    cx = np.cos(az) * np.sin(zd)
    cy = np.sin(az) * np.sin(zd)
    cz = np.cos(zd)
    return cx, cy, cz


def _az_zd_to_cx_cy(azimuth_deg, zenith_deg):
    cx, cy, _cz = _az_zd_to_cx_cy_cz(
        azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
    )
    return cx, cy


def _cx_cy_to_az_zd_deg(cx, cy):
    inner_sqrt = 1.0 - cx**2 - cy**2
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


def area_of_triangle(v0, v1, v2):
    """
    Returns the area of a triangle with vertices v0, v1, and v2.

    Parameters
    ----------
    v0 : np.array(3)
        First vertex of triangle.
    v1 : np.array(3)
        Second vertex of triangle.
    v2 : np.array(3)
        Third vertex of triangle.
    """
    l01 = v1 - v0
    l21 = v1 - v2
    return np.linalg.norm(np.cross(l01, l21)) / 2.0


def solid_angle_of_triangle_on_unitsphere(
    v0,
    v1,
    v2,
    delta_r=1e-6,
    delta_phi=np.deg2rad(60),
):
    """
    Returns the solid angle of a spherical triangle on the unit-sphere.

    According to girads theorem:
        solid angle = radius ** 2 * excess-angle
        excess-angle = (alpha + beta + gamma - pi)
        alpha: angle between line(v0, v1) and line(v0, v2)
        beta: angle between line(v1, v0) and line(v1, v2)
        gamma: angle between line(v2, v0) and line(v2, v1)
        v0, v1, v2 are the vertices of the triangle:

    Parameters
    ----------
    v0 : np.array(3)
        First vertex of triangle.
    v1 : np.array(3)
        Second vertex of triangle.
    v2 : np.array(3)
        Third vertex of triangle.
    """
    dot = np.dot
    norm = np.linalg.norm
    acos = np.arccos

    assert np.abs(norm(v0) - 1) <= delta_r
    assert np.abs(norm(v1) - 1) <= delta_r
    assert np.abs(norm(v2) - 1) <= delta_r

    alpha = angle_between(surface_tangent(v0, v1), surface_tangent(v0, v2))
    beta = angle_between(surface_tangent(v1, v0), surface_tangent(v1, v2))
    gamma = angle_between(surface_tangent(v2, v0), surface_tangent(v2, v1))

    excess_angle = alpha + beta + gamma - np.pi
    return excess_angle


def angle_between(a, b):
    """
    Returns the angle between the vectors a, and b.
    """
    dot = np.dot
    norm = np.linalg.norm
    acos = np.arccos
    return acos(dot(a, b) / (norm(a) * norm(b)))


def surface_tangent(a, b, delta_r=1e-6):
    """
    Returns the direction of the great-circle-arc which goes from point a to
    b and is located in point a.

    Parameters
    ----------
    a : vector dim 3
        Point 'a' on the unit-sphere.
    b : vector dim 3
        Point 'b' on the unit-sphere.

    Returns
    -------
    tangent : vector dim 3
        Direction-vector perpendicular to a and pointing in the
        great-circle-arc's direction towards b.
    """
    norm = np.linalg.norm
    assert np.abs(norm(a) - 1) <= delta_r
    assert np.abs(norm(b) - 1) <= delta_r

    ray_support = b
    ray_direction = a
    lam = ray_parameter_for_closest_distance_to_point(
        support_vector=ray_support,
        direction_vector=ray_direction,
        point=a,
    )
    closest_point = ray_support + lam * ray_direction
    tangent = closest_point - a
    assert np.abs(angle_between(tangent, a) - np.pi / 2) < 1e-6
    return tangent


def ray_parameter_for_closest_distance_to_point(
    support_vector, direction_vector, point
):
    d = np.dot(direction_vector, point)
    return d - np.dot(support_vector, direction_vector)
