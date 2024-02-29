import binning_utils
import scipy
from scipy import spatial
import numpy as np
import solid_angle_utils
import svg_cartesian_plot as svgplt
from sklearn.cluster import DBSCAN
import io
import triangle_mesh_io
import merlict
import corsika_primary
import spherical_coordinates


def make_vertices(
    num_vertices=1024,
    max_zenith_distance_rad=None,
):
    """
    Makes vertices on a unit-sphere using a Fibonacci-space.
    This is done to create mesh-faces of approximatly equal solid angles.

    Additional vertices are added at the horizon all around the azimuth to make
    shure the resulting mesh reaches the horizon at any azimuth.

    The Fibinacci-vertices and the horizon-ring-vertices are combined, while
    Fibonacci-vertices will be dropped when they are too close to existing
    vertices on the horizon-ring.

    Parameters
    ----------
    num_vertices : int
        A guidence for the number of verties in the mesh.
    max_zenith_distance_rad : float
        Vertices will only be put up to this zenith-distance.
        The ring-vertices will be put right at this zenith-distance.

    Returns
    -------
    vertices : numpy.array, shape(N, 3)
        The xyz-coordinates of the vertices.
    """
    PI = np.pi
    TAU = 2 * PI

    if max_zenith_distance_rad is None:
        max_zenith_distance_rad = corsika_primary.MAX_ZENITH_DISTANCE_RAD
    assert 0 < max_zenith_distance_rad <= np.pi / 2
    assert num_vertices > 0
    num_vertices = int(num_vertices)

    inner_vertices = binning_utils.sphere.fibonacci_space(
        size=num_vertices,
        max_zenith_distance_rad=max_zenith_distance_rad,
    )

    _hemisphere_solid_angle = 2.0 * np.pi
    _expected_num_faces = 2.0 * num_vertices
    _face_expected_solid_angle = _hemisphere_solid_angle / _expected_num_faces
    _face_expected_edge_angle_rad = np.sqrt(_face_expected_solid_angle)
    num_horizon_vertices = int(np.ceil(TAU / _face_expected_edge_angle_rad))

    horizon_vertices = []
    for az_rad in np.linspace(0, TAU, num_horizon_vertices, endpoint=False):
        uvec = np.array(
            spherical_coordinates.az_zd_to_cx_cy_cz(
                azimuth_rad=az_rad,
                zenith_rad=max_zenith_distance_rad,
            )
        )
        horizon_vertices.append(uvec)
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
    """
    Makes Delaunay-Triangle-faces for the given vertices. Only the x- and
    y coordinate are taken into account.

    Parameters
    ----------
    vertices : numpy.array
        The xyz-coordinates of the vertices.

    Returns
    -------
    delaunay_faces : numpy.array, shape(N, 3), int
        A list of N faces, where each face references the vertices it is made
        from.
    """
    delaunay = scipy.spatial.Delaunay(points=vertices[:, 0:2])
    delaunay_faces = delaunay.simplices
    return delaunay_faces


def estimate_vertices_to_faces_map(faces, num_vertices):
    """
    Parameters
    ----------
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    num_vertices : int
        The total number of vertices in the mesh

    Returns
    -------
    nn : dict of lists
        A dict with an entry for each vertex referencing the faces it is
        connected to.
    """
    nn = {}
    for iv in range(num_vertices):
        nn[iv] = set()

    for iface, face in enumerate(faces):
        for iv in face:
            nn[iv].add(iface)

    out = {}
    for key in nn:
        out[key] = list(nn[key])
    return out


def estimate_solid_angles(vertices, faces, geometry="spherical"):
    """
    For a given hemispherical mesh defined by vertices and faces, calculate the
    solid angle of each face.

    Parameters
    ----------
    vertices : numpy.array, shape(M, 3), float
        The xyz-coordinates of the M vertices.
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    geometry : str, default="spherical"
        Whether to apply "spherical" or "flat" geometry. Where "flat" geometry
        is only applicable for small faces.

    Returns
    -------
    solid : numpy.array, shape=(N, ), float
        The individual solid angles of the N faces in the mesh
    """
    solid = np.nan * np.ones(len(faces))
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

        solid[i] = face_solid_angle
    return solid


def cluster(
    vertices,
    eps,
    min_samples,
):
    assert eps > 0
    assert min_samples > 0
    if len(vertices) == 0:
        return np.array([])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(vertices)
    return dbscan.labels_


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


class Grid:
    """
    A hemispherical grid with a Fibonacci-spacing.
    """

    def __init__(
        self,
        num_vertices,
        max_zenith_distance_rad=None,
    ):
        """
        Parameters
        ----------
        num_vertices : int
            A guideline for the number of vertices in the grid's mesh.
            See make_vertices().
        max_zenith_distance_rad : float
            Vertices will only be put up to this zenith-distance.
            The ring-vertices will be put right at this zenith-distance.
        """
        if max_zenith_distance_rad is None:
            max_zenith_distance_rad = corsika_primary.MAX_ZENITH_DISTANCE_RAD

        self._init_num_vertices = int(num_vertices)
        self.max_zenith_distance_rad = float(max_zenith_distance_rad)
        self.vertices = make_vertices(
            num_vertices=self._init_num_vertices,
            max_zenith_distance_rad=self.max_zenith_distance_rad,
        )
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

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        """
        Returns the index of the face hit at direction
        (azimuth_rad, zenith_rad).
        """
        return self.faces_tree.query_azimuth_zenith(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )

    def query_cx_cy(self, cx, cy):
        """
        Returns the index of the face hit at direction (cx, cy).
        """
        return self.faces_tree.query_cx_cy(cx=cx, cy=cy)

    def query(self, direction_unit_vector):
        """
        Returns the index of the face hit by the direction_unit_vector.
        """
        return self.faces_tree.query(
            direction_unit_vector=direction_unit_vector
        )

    def plot(slef, path):
        """
        Writes a plot with the grid's faces to path.
        """
        plot(self.vertices, self.faces, path)

    def __repr__(self):
        return "{:s}(num_vertices={:d})".format(
            self.__class__.__name__,
            self._init_num_vertices,
        )


def make_hemisphere_obj(vertices, faces, mtlkey="sky"):
    """
    Makes an object-wavefron dict() from the hemispherical mesh defined by
    vertices and faces.

    Parameters
    ----------
    vertices : numpy.array, shape(M, 3), float
        The xyz-coordinates of the M vertices. The vertices are expected to be
        on the unit-sphere.
    faces : numpy.array, shape(N, 3), int
        A list of N faces referencing their vertices.
    mtlkey : str, default="sky"
        Key indicating the first and only material in the object-wavefront.

    Returns
    -------
    obj : dict representing an object-wavefront
        Includes vertices, vertex-normals, and materials ('mtl's) with faces.
    """
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
    """
    An acceleration structure to allow fast queries for rays hitting a
    mesh defined by vertices and faces.
    """

    def __init__(self, vertices, faces):
        """
        Parameters
        ----------
        vertices : numpy.array, shape(M, 3), float
            The xyz-coordinates of the M vertices. The vertices are expected
            to be on the unit-sphere.
        faces : numpy.array, shape(N, 3), int
            A list of N faces referencing their vertices.
        """
        scenery_py = make_merlict_scenery_py(vertices=vertices, faces=faces)
        self._tree = merlict.compile(sceneryPy=scenery_py)

    def _make_probing_ray(self, direction_unit_vector):
        if len(direction_unit_vector.shape) == 2:
            num_vectors = direction_unit_vector.shape[0]
            assert direction_unit_vector.shape[1] == 3
            rays = merlict.ray.init(num_vectors)
            rays["support.x"] = np.zeros(num_vectors)
            rays["support.y"] = np.zeros(num_vectors)
            rays["support.z"] = np.zeros(num_vectors)
            rays["direction.x"] = direction_unit_vector[:, 0]
            rays["direction.y"] = direction_unit_vector[:, 1]
            rays["direction.z"] = direction_unit_vector[:, 2]
            return rays

        elif len(direction_unit_vector.shape) == 1:
            assert 0.99 <= np.linalg.norm(direction_unit_vector) <= 1.01
            ray = merlict.ray.init(1)
            ray["support.x"] = 0.0
            ray["support.y"] = 0.0
            ray["support.z"] = 0.0
            ray["direction.x"] = direction_unit_vector[0]
            ray["direction.y"] = direction_unit_vector[1]
            ray["direction.z"] = direction_unit_vector[2]
            return ray
        else:
            raise ValueError(
                "vector must either have shape (3,) or shape (N,3)"
            )

    def query_azimuth_zenith(self, azimuth_rad, zenith_rad):
        direction_unit_vector = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        return self.query(direction_unit_vector=direction_unit_vector)

    def query_cx_cy(self, cx, cy):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        direction_unit_vector = np.c_[cx, cy, cz]
        return self.query(direction_unit_vector=direction_unit_vector)

    def query(self, direction_unit_vector):
        assert len(direction_unit_vector.shape) == 2
        num_vectors = direction_unit_vector.shape[0]
        assert direction_unit_vector.shape[1] == 3

        _hits, _intersecs = self._tree.query_intersection(
            self._make_probing_ray(direction_unit_vector)
        )

        face_ids = np.zeros(num_vectors, dtype=int)
        face_ids[np.logical_not(_hits)] = -1
        face_ids[_hits] = _intersecs["geometry_id.face"][_hits]
        return face_ids


class Mask:
    """
    A mask for the hemispherical grid to mark certain faces/cells.
    """

    def __init__(self, grid):
        self.grid = grid
        self.faces = set()

    def solid_angle(self):
        """
        Returns
        -------
        solid_angle : float
            The total solid angle covered by all masked faces in the
            hemispherical grid.
        """
        total_sr = 0.0
        for iface in self.faces:
            total_sr += self.grid.faces_solid_angles[iface]
        return total_sr

    def append_azimuth_zenith(self, azimuth_rad, zenith_rad, half_angle_rad):
        """
        Marks all faces in a cone at direction (azimuth_rad, zenith_rad) and
        opening half_angle_rad.
        """
        direction_unit_vector = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad, zenith_rad=zenith_rad
        )
        return self.append(
            direction_unit_vector=direction_unit_vector,
            half_angle_rad=half_angle_rad,
        )

    def append_cx_cy(self, cx, cy, half_angle_rad):
        cz = spherical_coordinates.restore_cz(cx=cx, cy=cy)
        assert 0.0 <= cz <= 1.0
        direction_unit_vector = np.array([cx, cy, cz])
        self.append(
            direction_unit_vector=direction_unit_vector,
            half_angle_rad=half_angle_rad,
        )

    def append(self, direction_unit_vector, half_angle_rad):
        assert half_angle_rad >= 0
        assert 0.99 <= np.linalg.norm(direction_unit_vector) <= 1.01

        third_neighbor_angle_rad = np.max(
            self.grid.vertices_tree.query(x=direction_unit_vector, k=3)[0]
        )

        query_angle_rad = np.max([half_angle_rad, third_neighbor_angle_rad])

        vidx_in_cone = self.grid.vertices_tree.query_ball_point(
            x=direction_unit_vector,
            r=query_angle_rad,
        )

        for vidx in vidx_in_cone:
            faces_touching_vidx = self.grid.vertices_to_faces_map[vidx]
            for face in faces_touching_vidx:
                self.faces.add(face)

    def plot(self, path):
        """
        Writes a plot with the grid's faces to path.
        """
        faces_mask = np.zeros(shape=len(self.grid.faces), dtype=np.int)
        faces_mask[self.faces] = 1
        plot(
            vertices=self.grid.vertices,
            faces=self.grid.faces,
            faces_mask=faces_mask,
            path=path,
        )

    def __repr__(self):
        return "{:s}(grid=Grid(num_vertices{:d})".format(
            self.__class__.__name__,
            self.mesh._init_num_vertices,
        )
