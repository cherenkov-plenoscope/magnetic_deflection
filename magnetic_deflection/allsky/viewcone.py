import numpy as np
import homogeneous_transformation as htr
import atmospheric_cherenkov_response as acr


def make_ring(half_angle_deg, fn=6, endpoint=True):
    verts = []
    sin = np.sin
    cos = np.cos
    zd = np.deg2rad(half_angle_deg)

    for az in np.linspace(0.0, 2 * np.pi, fn, endpoint=endpoint):
        xx = sin(zd) * cos(az)
        yy = sin(zd) * sin(az)
        zz = cos(zd)
        verts.append([xx, yy, zz])
    return np.array(verts)


def limit_zenith_distance(vertices_uxyz, max_zenith_distance_deg):
    out = []
    for v in vertices_uxyz:
        az_deg, zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=vertices_uxyz[0],
            cy=vertices_uxyz[1],
        )
        if zd_deg > max_zenith_distance_deg:
            zd_deg = max_zenith_distance_deg
        out_cx_cy_cy = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=az_deg,
            zenith_deg=zd_deg,
        )
        out.append(np.array(out_cx_cy_cy))
    return np.array(out)


def rotate(vertices_uxyz, azimuth_deg, zenith_deg, mount="cable_robot_mount"):
    _rot_civil = acr.pointing.make_civil_rotation_of_principal_aperture_plane(
        pointing={"azimuth_deg": azimuth_deg, "zenith_deg": zenith_deg},
        mount=mount,
    )
    _trafo_civil = {"pos": [0, 0, 0], "rot": _rot_civil}
    rot_vertices_uxyz = htr.transform_orientation(
        t=htr.compile(_trafo_civil),
        d=vertices_uxyz,
    )
    return rot_vertices_uxyz
