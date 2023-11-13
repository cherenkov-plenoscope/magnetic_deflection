import binning_utils
import scipy
from scipy import spatial
import numpy as np
import solid_angle_utils
import svg_cartesian_plot as svgplt
from .. import spherical_coordinates
import io
import triangle_mesh_io
import merlict


def make_vertices(num_vertices=1024):
    max_zenith_distance_deg = int(90)
    assert num_vertices > 0
    num_vertices = int(num_vertices)

    inner_vertices = binning_utils.sphere.fibonacci_space(
        size=num_vertices,
        max_zenith_distance_rad=np.deg2rad(max_zenith_distance_deg),
    )

    _hemisphere_solid_angle = 2.0 * np.pi
    _expected_num_faces = 2.0 * num_vertices
    _face_expected_solid_angle = _hemisphere_solid_angle / _expected_num_faces
    _face_expected_edge_angle_rad = 2.0 * np.sqrt(_face_expected_solid_angle)
    _face_expected_edge_angle_deg = np.rad2deg(_face_expected_edge_angle_rad)
    num_horizon_vertices = int(np.ceil(360.0 / _face_expected_edge_angle_deg))

    horizon_vertices = []
    for az_deg in np.linspace(0, 360, num_horizon_vertices, endpoint=False):
        horizon_vertices.append(
            [np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg)), 0.0]
        )
    horizon_vertices = np.array(horizon_vertices)

    vertices = []

    _horizon_vertices_tree = scipy.spatial.cKDTree(data=horizon_vertices)
    for inner_vertex in inner_vertices:
        delta_rad, vidx = _horizon_vertices_tree.query(inner_vertex)

        if delta_rad > _face_expected_edge_angle_rad:
            vertices.append(inner_vertex)

    for horizon_vertex in horizon_vertices:
        vertices.append(horizon_vertex)

    return np.array(vertices)


def make_faces(vertices):
    delaunay = scipy.spatial.Delaunay(points=vertices[:, 0:2])
    delaunay_faces = delaunay.simplices
    return delaunay_faces


def estimate_vertices_to_faces_map(faces, num_vertices):
    nn = {}
    for iv in range(num_vertices):
        nn[iv] = set()

    for iface, face in enumerate(faces):
        for iv in face:
            nn[iv].add(iface)
    return nn


def estimate_solid_angles(vertices, faces, geometry="spherical"):
    sol = np.nan * np.ones(len(faces))
    for i in range(len(faces)):
        face = faces[i]
        if geometry == "spherical":
            face_solid_angle = solid_angle_utils.triangle.solid_angle(
                v0=vertices[face[0]],
                v1=vertices[face[1]],
                v2=vertices[face[2]],
            )
        elif geometry == "flat":
            face_solid_angle = (
                solid_angle_utils.triangle._area_of_flat_triangle(
                    v0=vertices[face[0]],
                    v1=vertices[face[1]],
                    v2=vertices[face[2]],
                )
            )
        else:
            raise ValueError(
                "Expected geometry to be either 'flat' or 'spherical'."
            )

        sol[i] = face_solid_angle
    return sol


def plot(vertices, faces, path, faces_mask=None):
    fig = svgplt.Fig(cols=1080, rows=1080)
    ax = svgplt.hemisphere.Ax(fig=fig)
    mesh_look = svgplt.hemisphere.init_mesh_look(
        num_faces=len(faces),
        fill=svgplt.color.css("RoyalBlue"),
        fill_opacity=0.5,
    )
    if faces_mask:
        for i in range(len(self.faces)):
            if faces_mask[i]:
                color = svgplt.color.css("red")
            else:
                color = svgplt.color.css("blue")
            mesh_look["faces_fill"][i] = color
    svgplt.hemisphere.ax_add_mesh(
        ax=ax,
        vertices=vertices,
        faces=faces,
        max_radius=1.0,
        **mesh_look,
    )
    svgplt.hemisphere.ax_add_grid(ax=ax)
    svgplt.fig_write(fig=fig, path=path)


class Mesh:
    def __init__(self, num_vertices):
        self._init_num_vertices = int(num_vertices)
        self.vertices = make_vertices(num_vertices=self._init_num_vertices)
        self.vertices_tree = scipy.spatial.cKDTree(data=self.vertices)
        self.faces = make_faces(vertices=self.vertices)
        self.vertices_to_faces_map = estimate_vertices_to_faces_map(
            faces=self.faces, num_vertices=len(self.vertices)
        )
        self.faces_solid_angles = estimate_solid_angles(
            vertices=self.vertices,
            faces=self.faces,
        )
        self.faces_tree = Tree(vertices=self.vertices, faces=self.faces)

    def query_azimuth_zenith(self, azimuth_deg, zenith_deg):
        return self.faces_tree.query_azimuth_zenith(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )

    def query_cx_cy(self, cx, cy):
        return self.faces_tree.query_cx_cy(cx=cx, cy=cy)

    def query(self, direction_unit_vector):
        return self.faces_tree.query(
            direction_unit_vector=direction_unit_vector
        )

    def plot(slef, path):
        plot(self.vertices, self.faces, path)

    def __repr__(self):
        return "{:s}(num_vertices={:d})".format(
            self.__class__.__name__,
            self._init_num_vertices,
        )


def make_hemisphere_obj(vertices, faces, mtlkey="sky"):
    obj = triangle_mesh_io.obj.init()
    for vertex in vertices:
        obj["v"].append(vertex)
        # all vertices are on a sphere
        # so the vertex is parallel to its surface-normal.
        obj["vn"].append(vertex)
    obj["mtl"] = {}
    obj["mtl"][mtlkey] = []
    for face in faces:
        obj["mtl"][mtlkey].append({"v": face, "vn": face})
    return obj


def make_merlict_scenery_py(vertices, faces):
    scenery_py = merlict.scenery.init(default_medium="vacuum")
    scenery_py["geometry"]["objects"]["hemisphere"] = make_hemisphere_obj(
        vertices=vertices, faces=faces, mtlkey="sky"
    )
    scenery_py["materials"]["surfaces"][
        "perfect_absorber"
    ] = merlict.materials.surfaces.init(key="perfect_absorber")

    scenery_py["geometry"]["relations"]["children"].append(
        {
            "id": 0,
            "pos": [0, 0, 0],
            "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
            "obj": "hemisphere",
            "mtl": {"sky": "abc"},
        }
    )
    scenery_py["materials"]["boundary_layers"]["abc"] = {
        "inner": {"medium": "vacuum", "surface": "perfect_absorber"},
        "outer": {"medium": "vacuum", "surface": "perfect_absorber"},
    }
    return scenery_py


class Tree:
    def __init__(self, vertices, faces):
        scenery_py = make_merlict_scenery_py(vertices=vertices, faces=faces)
        self._tree = merlict.compile(sceneryPy=scenery_py)

    def _make_probing_ray(self, direction_unit_vector):
        assert 0.99 <= np.linalg.norm(direction_unit_vector) <= 1.01
        ray = merlict.ray.init(1)
        ray["support.x"] = 0.0
        ray["support.y"] = 0.0
        ray["support.z"] = 0.0
        ray["direction.x"] = direction_unit_vector[0]
        ray["direction.y"] = direction_unit_vector[1]
        ray["direction.z"] = direction_unit_vector[2]
        return ray

    def query_azimuth_zenith(self, azimuth_deg, zenith_deg):
        direction_unit_vector = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        return self.query(direction_unit_vector=direction_unit_vector)

    def query_cx_cy(self, cx, cy):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        assert 0.0 <= cz <= 1.0
        direction_unit_vector = np.array([cx, cy, cz])
        return self.query(direction_unit_vector=direction_unit_vector)

    def query(self, direction_unit_vector):
        _hits, _intersecs = self._tree.query_intersection(
            self._make_probing_ray(direction_unit_vector)
        )
        assert len(_hits) == 1
        assert len(_intersecs) == 1

        hit = _hits[0]
        intersec = _intersecs[0]
        face_id = -1

        if hit:
            face_id = intersec["geometry_id.face"]
            assert 0.95 < intersec["distance_of_ray"] <= (1.0 + 1e-6)

        return face_id


class Mask:
    def __init__(self, mesh):
        self.mesh = mesh
        self.faces = set()

    def solid_angle(self):
        total_sr = 0.0
        for iface in self.faces:
            total_sr += self.mesh.faces_solid_angles[iface]
        return total_sr
        """
        Returns
        -------
        solid_angle : float
            The total solid angle covered by all masked faces in the
            hemispherical grid.
        """

    def append_azimuth_zenith(self, azimuth_deg, zenith_deg, half_angle_deg):
        direction_unit_vector = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        return self.append(
            direction_unit_vector=direction_unit_vector,
            half_angle_deg=half_angle_deg,
        )

    def append_cx_cy(self, cx, cy, half_angle_deg):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        assert 0.0 <= cz <= 1.0
        direction_unit_vector = np.array([cx, cy, cz])
        self.append(
            direction_unit_vector=direction_unit_vector,
            half_angle_deg=half_angle_deg,
        )

    def append(self, direction_unit_vector, half_angle_deg):
        assert half_angle_deg >= 0
        assert 0.99 <= np.linalg.norm(direction_unit_vector) <= 1.01
        half_angle_rad = np.deg2rad(half_angle_deg)

        third_neighbor_angle_rad = np.max(
            self.mesh.vertices_tree.query(x=direction_unit_vector, k=3)[0]
        )

        query_angle_rad = np.max([half_angle_rad, third_neighbor_angle_rad])

        vidx_in_cone = self.mesh.vertices_tree.query_ball_point(
            x=direction_unit_vector,
            r=query_angle_rad,
        )

        for vidx in vidx_in_cone:
            faces_touching_vidx = self.mesh.vertices_to_faces_map[vidx]
            for face in faces_touching_vidx:
                self.faces.add(face)

    def plot(self, path):
        faces_mask = np.zeros(shape=len(self.mesh.faces), dtype=np.int)
        faces_mask[self.faces] = 1
        plot(
            vertices=self.mesh.vertices,
            faces=self.mesh.faces,
            faces_mask=faces_mask,
            path=path,
        )

    def __repr__(self):
        return "{:s}(mesh=Mesh(num_vertices{:d})".format(
            self.__class__.__name__,
            self.mesh._init_num_vertices,
        )
