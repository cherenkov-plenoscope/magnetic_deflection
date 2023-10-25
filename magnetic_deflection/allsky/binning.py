import scipy
from .. import spherical_coordinates
from scipy import spatial
import binning_utils
import numpy as np
import svg_cartesian_plot as splt
import copy


class Binning:
    def __init__(self, config):
        """
        A binning in directions and energy.
        The direction-binning is in a sphere.
        The energy binning is a geomspace.

        Compiles the Binning from the config-dict read from work_dir/config.
        """
        self.config = copy.deepcopy(config)
        _energy_bin_edges = np.geomspace(
            self.config["energy"]["start_GeV"],
            self.config["energy"]["stop_GeV"],
            self.config["energy"]["num_bins"] + 1,
        )
        self.energy = binning_utils.Binning(bin_edges=_energy_bin_edges)
        self.max_zenith_distance_deg = 90
        centers = binning_utils.sphere.fibonacci_space(
            size=self.config["direction"]["num_bins"],
            max_zenith_distance_rad=np.deg2rad(self.max_zenith_distance_deg),
        )
        _hemisphere_solid_angle = 2.0 * np.pi
        _expected_num_delaunay_faces = (
            2.0 * self.config["direction"]["num_bins"]
        )
        self.direction_voronoi_face_expected_solid_angle = (
            _hemisphere_solid_angle / _expected_num_delaunay_faces
        )
        self.direction = scipy.spatial.cKDTree(data=centers)
        self.horizon_vertices = []
        for az_deg in np.linspace(0, 360, 36, endpoint=False):
            self.horizon_vertices.append([np.cos(az_deg), np.sin(az_deg), 0.0])
        self.horizon_vertices = np.array(self.horizon_vertices)

    def query(self, azimuth_deg, zenith_deg, energy_GeV):
        """
        Finds the closest bin for a given direction and energy.
        The direction is given in (azimuth_deg, zenith_deg).

        Parameters
        ----------
        azimuth_deg : float
            Azimuth angle to find the closest match for in DEG.
        zenith_deg : float
            Zenith angle to find the closest match for in DEG.
        energy_GeV : float
            Energy to find the closest match for in GeV.

        Retruns
        -------
        (d_angle_deg, e_distance_GeV), (dbin, ebin)

        d_angle_deg : float
            Angle to the closest direction-bin-center.
        e_distance_GeV : float
            Absolute energy-distance to the closest bin-edge in energy.
        dbin : int
            Index of the closest direction-bin.
        ebin : int
            Index of the closest energy-bin.
        """
        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        pointing = np.array([cx, cy, cz])
        p_angle_rad, dbin = self.direction.query(pointing)
        ebin = np.digitize(energy_GeV, self.energy["edges"]) - 1

        ee = self.energy["edges"] / energy_GeV
        ee[ee < 1] = 1 / ee[ee < 1]
        eclose = np.argmin(ee)

        e_distance_GeV = np.abs(self.energy["edges"][eclose] - energy_GeV)

        d_angle_deg = np.rad2deg(p_angle_rad)
        return (d_angle_deg, e_distance_GeV), (dbin, ebin)

    def query_ball(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_start_GeV,
        energy_stop_GeV,
    ):
        """
        Finds the bins within a given direction-cone and energy-range.

        Parameters
        ----------
        azimuth_deg : float
            Direction's azimuth in DEG.
        zenith_deg : float
            Direction's zenith in DEG.
        half_angle_deg : float
            Cone's half angle (on the sky-dome) which encircles the direction.
            All direction-bins within this cone will be returned.
        energy_start_GeV : float
            Start of energy-range in GeV.
        energy_stop_GeV : float
            Stop of energy-range in GeV.

        Returns
        -------
        (dd, ee)

        dd : np.array
            Indices of direction-bins which are within the query's radius.
        ee : np.array
            Indices of energy-bins which are within the query's radius.
        """
        assert energy_start_GeV > 0
        assert energy_stop_GeV > 0
        assert energy_stop_GeV > energy_start_GeV
        assert half_angle_deg >= 0

        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        pointing = np.array([cx, cy, cz])
        dd = self.direction.query_ball_point(
            x=pointing, r=np.deg2rad(half_angle_deg)
        )
        dd = np.array(dd)

        ebin_start = np.digitize(energy_start_GeV, self.energy["edges"]) - 1
        ebin_stop = np.digitize(energy_stop_GeV, self.energy["edges"]) - 1

        ee = set()

        if ebin_start >= 0 and ebin_start < self.energy["num"]:
            ee.add(ebin_start)
        if ebin_stop >= 0 and ebin_stop < self.energy["num"]:
            ee.add(ebin_stop)

        ee = np.array(list(ee))
        if len(ee) == 2:
            ee = np.arange(min(ee), max(ee) + 1)

        return (dd, ee)

    def _project_direction_bin_centers_in_xy_plane(self):
        direction_bin_centers = self.direction.data.copy()
        return direction_bin_centers[:, 0:2]

    def direction_delaunay_mesh(self):
        direction_bin_centers = self.direction.data.copy()
        direction_horizon = self.horizon_vertices
        vertices = np.vstack([direction_bin_centers, direction_horizon])
        delaunay = scipy.spatial.Delaunay(points=vertices[:, 0:2])
        delaunay_faces = delaunay.simplices
        return vertices, delaunay_faces

    def direction_delaunay_mesh_solid_angles(self):
        vertices, faces = self.direction_delaunay_mesh()
        sol = np.zeros(len(faces))
        for i in range(len(faces)):
            face = faces[i]
            face_solid_angle = spherical_coordinates.solid_angle_of_triangle_on_unitsphere(
                v0=vertices[face[0]],
                v1=vertices[face[1]],
                v2=vertices[face[2]],
            )
            sol[i] = face_solid_angle
        return sol

    def direction_num_bins(self):
        return len(self.direction.data)

    def is_valid_dbin_ebin(self, dbin, ebin):
        dvalid = 0 <= dbin < self.config["direction"]["num_bins"]
        evalid = 0 <= ebin < self.config["energy"]["num_bins"]
        return dvalid and evalid

    def plot(self, path, fill="blue"):
        fig = splt.Fig(cols=1080, rows=1080)
        ax = splt.hemisphere.Ax(fig=fig)

        max_par_zd_deg = self.config["direction"][
            "particle_max_zenith_distance_deg"
        ]
        splt.shapes.ax_add_circle(
            ax=ax,
            xy=[0, 0],
            radius=np.sin(np.deg2rad(max_par_zd_deg)),
            stroke=splt.color.css("red"),
        )
        max_cer_zd_deg = self.config["direction"][
            "cherenkov_max_zenith_distance_deg"
        ]
        splt.shapes.ax_add_circle(
            ax=ax,
            xy=[0, 0],
            radius=np.sin(np.deg2rad(max_cer_zd_deg)),
            stroke=splt.color.css("blue"),
        )

        vertices, faces = self.direction_voronoi_mesh()
        mesh_look = splt.hemisphere.init_mesh_look(
            num_faces=len(faces),
            fill=splt.color.css("RoyalBlue"),
            fill_opacity=0.5,
        )

        splt.hemisphere.ax_add_mesh(
            ax=ax,
            vertices=vertices,
            faces=faces,
            max_radius=1.0,
            **mesh_look,
        )

        splt.hemisphere.ax_add_grid(ax=ax)

        splt.fig_write(fig=fig, path=path)

    def __repr__(self):
        out = "{:s}(num bins: energy={:d}, direction={:d})".format(
            self.__class__.__name__,
            self.config["energy"]["num_bins"],
            self.config["direction"]["num_bins"],
        )
        return out
