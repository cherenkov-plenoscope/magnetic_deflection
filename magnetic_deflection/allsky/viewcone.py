import numpy as np
import homogeneous_transformation as htr
import atmospheric_cherenkov_response as acr
import spherical_coordinates


def make_ring(half_angle_rad, fn=6, endpoint=True):
    verts = []
    sin = np.sin
    cos = np.cos
    zd = half_angle_rad

    for az in np.linspace(0.0, 2 * np.pi, fn, endpoint=endpoint):
        xx = sin(zd) * cos(az)
        yy = sin(zd) * sin(az)
        zz = cos(zd)
        verts.append([xx, yy, zz])
    return np.array(verts)


def limit_zenith_distance(vertices_uxyz, max_zenith_distance_rad):
    out = []
    for v in vertices_uxyz:
        az_rad, zd_rad = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=v[0],
            cy=v[1],
            cz=v[2],
        )
        if zd_rad > max_zenith_distance_rad:
            zd_rad = max_zenith_distance_rad
        out_cx_cy_cy = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=az_rad,
            zenith_rad=zd_rad,
        )
        out.append(np.array(out_cx_cy_cy))
    return np.array(out)


def rotate(vertices_uxyz, azimuth_rad, zenith_rad, mount="cable_robot_mount"):
    _rot_civil = acr.pointing.make_civil_rotation_of_principal_aperture_plane(
        pointing={
            "azimuth_deg": np.rad2deg(azimuth_rad),
            "zenith_deg": np.rad2deg(zenith_rad),
        },
        mount=mount,
    )
    _trafo_civil = {"pos": [0, 0, 0], "rot": _rot_civil}
    rot_vertices_uxyz = htr.transform_orientation(
        t=htr.compile(_trafo_civil),
        d=vertices_uxyz,
    )
    return rot_vertices_uxyz
