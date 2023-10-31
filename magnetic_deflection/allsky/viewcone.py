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
