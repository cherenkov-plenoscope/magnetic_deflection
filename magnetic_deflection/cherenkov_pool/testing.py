import numpy as np
import homogeneous_transformation as ht
import corsika_primary
from .. import spherical_coordinates


def draw_xy_in_disc(prng, x, y, size, radius=1e2):
    radii = prng.uniform(low=0.0, high=radius, size=size)
    phis = prng.uniform(low=0.0, high=2 * np.pi, size=size)
    px = np.zeros(size)
    py = np.zeros(size)
    for i in range(size):
        px[i] = radii[i] * np.cos(phis[i]) + x
        py[i] = radii[i] * np.sin(phis[i]) + y
    return {"x": px, "y": py}, {"radii": radii, "phis": phis}


def draw_direction(prng, max_zenith_distance):
    (
        az,
        zd,
    ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
        prng=prng,
        azimuth_rad=0.0,
        zenith_rad=0.0,
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=max_zenith_distance,
        max_zenith_rad=np.deg2rad(180),
        max_iterations=1000 * 1000,
    )
    ppp = np.array(
        spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=np.rad2deg(az),
            zenith_deg=np.rad2deg(zd),
        )
    )
    return ppp


def draw_cxcy_in_cone(prng, cx, cy, size, half_angle):
    cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
    pointing = np.array([cx, cy, cz])
    unit_z = np.array([0.0, 0.0, 1.0])
    hypo = 1.0
    height = hypo * np.cos(half_angle)
    radius = hypo * np.sin(half_angle)
    emission_point = np.array([0.0, 0.0, height])
    lf, _ = draw_xy_in_disc(prng=prng, x=0.0, y=0.0, size=size, radius=radius)

    theta = np.zeros(size)
    ppp = np.zeros(shape=(size, 3))
    for i in range(size):
        disc_point = np.array([lf["x"][i], lf["y"][i], 0.0])
        ddd = -disc_point + emission_point
        ddn = ddd / np.linalg.norm(ddd)
        theta[i] = spherical_coordinates._angle_between_vectors_rad(
            ddn, unit_z
        )
        ppp[i] = ddn

    rotaxis = np.cross(pointing, unit_z)
    rotangle = -1.0 * spherical_coordinates._angle_between_vectors_rad(
        unit_z, pointing
    )

    trafo_civil = {
        "pos": np.array([0.0, 0.0, 0.0]),
        "rot": {
            "repr": "axis_angle",
            "axis": rotaxis,
            "angle_deg": np.rad2deg(rotangle),
        },
    }
    trafo = ht.compile(trafo_civil)

    rrr = ht.transform_orientation(trafo, ppp)

    return {"cx": rrr[:, 0], "cy": rrr[:, 1]}, {"theta": theta}


def draw_dummy_light_field(
    prng,
    size,
    radius=1e2,
    x=0.0,
    y=0.0,
    half_angle=15.0,
    cx=0.0,
    cy=0.0,
    t=0.0,
    t_std=5e-9,
):
    lf = {}
    truth = {}

    _lf, _truth = draw_xy_in_disc(
        prng=prng,
        x=x,
        y=y,
        size=size,
        radius=radius,
    )
    lf.update(_lf)
    truth.update(_truth)

    _lf, _truth = draw_cxcy_in_cone(
        prng=prng, cx=cx, cy=cy, size=size, half_angle=half_angle
    )
    lf.update(_lf)
    truth.update(_truth)

    lf["t"] = prng.normal(loc=t, scale=t_std, size=size)

    return lf, truth
