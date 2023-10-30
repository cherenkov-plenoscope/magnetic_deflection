import numpy as np
import homogeneous_transformation as htr
import atmospheric_cherenkov_response as acr
import binning_utils
import scipy
from scipy import spatial
from .. import spherical_coordinates
import optic_object_wavefronts as oow


def init(
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    num_vertices=64,
    mount="cable_robot_mount",
):
    assert half_angle_deg > 0.0

    vc = {}
    vc["azimuth_deg"] = float(azimuth_deg)
    vc["zenith_deg"] = float(zenith_deg)
    vc["half_angle_deg"] = float(half_angle_deg)
    vc["mount"] = str(mount)

    vc["vertices_wrt_zenith"], vc["faces"] = make_mesh(
        half_angle_deg=self.half_angle_deg, num_vertices=num_vertices
    )

    _rot_civil = acr.pointing.make_civil_rotation_of_principal_aperture_plane(
        pointing={
            "azimuth_deg": vc["azimuth_deg"],
            "zenith_deg": vc["zenith_deg"],
        },
        mount=vc["mount"],
    )
    _trafo_civil = {"pos": [0, 0, 0], "rot": _rot_civil}

    vc["vertices"] = htr.transform_orientation(
        t=htr.compile(_trafo_civil),
        d=vc["vertices_wrt_zenith"],
    )



def make_mesh(half_angle_deg, fn=2):
    """
    Returns a mesh (vertices, faces) with all vertices on the unitsphere.
    The mesh represents the spherical surface of a viewcone and tiles
    it with roughly equally sized triangular faces.
    The cone is not round, but instead is tiled together from triangular faces
    in a regular grid.

    Parameters
    ----------
    half_angle_deg : float
        The viewcone's half-angle in DEG.
    fn : int (default=2)
        Resolution of the mesh. The number of triangular faces on along the
        viewcone's diagonal. E.g. fn=2 will return six triangular faces
        in a hexagonal viewcone.
    """
    assert half_angle_deg >= 0.0
    _mesh = oow.primitives.spherical_cap_pixels.init(
        outer_radius=np.sin(np.deg2rad(half_angle_deg)),
        curvature_radius=1.0,
        fn_hex_grid=fn,
    )

    for vkey in _mesh["vertices"]:
        vert = _mesh["vertices"][vkey]
        overt = np.array([vert[0], vert[1], -vert[2]])
        _mesh["vertices"][vkey] = overt

    _mesh = oow.mesh.translate(mesh=_mesh, v=np.array([0, 0, 1]))
    _off_mesh = oow.export.reduce_mesh_to_off(_mesh)

    vertices = np.array(_off_mesh["v"])
    faces = np.array(_off_mesh["f"])

    for vertex in vertices:
        assert 0.99 < np.linalg.norm(vertex) < 1.01

    return vertices, faces
