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
        self.energy = binning_utils.Binning(
            _init_energy_bin_edges(
                start_GeV=self.config["energy"]["start_GeV"],
                stop_GeV=self.config["energy"]["stop_GeV"],
                num_bins=self.config["energy"]["num_bins"],
            )
        )
        centers = binning_utils.sphere.fibonacci_space(
            size=self.config["direction"]["num_bins"],
            max_zenith_distance_rad=90,
        )
        self.max_zenith_distance_deg = np.mod(np.rad2deg(90), 360)
        self.direction = scipy.spatial.cKDTree(data=centers)

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
        (p_angle_deg, e_distance_GeV), (pbin, ebin)

        p_angle_deg : float
            Angle to the closest direction-bin-center.
        e_distance_GeV : float
            Absolute energy-distance to the closest bin-edge in energy.
        pbin : int
            Index of the closest direction-bin.
        ebin : int
            Index of the closest energy-bin.
        """
        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        pointing = np.array([cx, cy, cz])
        p_angle_rad, pbin = self.direction.query(pointing)
        ebin = np.digitize(energy_GeV, self.energy["edges"]) - 1

        ee = self.energy["edges"] / energy_GeV
        ee[ee < 1] = 1 / ee[ee < 1]
        eclose = np.argmin(ee)

        e_distance_GeV = np.abs(self.energy["edges"][eclose] - energy_GeV)

        p_angle_deg = np.rad2deg(p_angle_rad)
        return (p_angle_deg, e_distance_GeV), (pbin, ebin)

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
        (pp, ee)

        pp : np.array
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
        pp = self.direction.query_ball_point(
            x=pointing, r=np.deg2rad(half_angle_deg)
        )
        pp = np.array(pp)

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

        return (pp, ee)

    def direction_voronoi_mesh(self):
        """
        Returns a mesh of vertices and faces which represent the
        voronoi-diagram of the directional binning.

        This is only a projection in x and y of the hemisphere.
        """
        direction_bin_centers = self.direction.data.copy()
        points_xy = direction_bin_centers[:, 0:2]
        voro = scipy.spatial.Voronoi(points=points_xy)

        # assign the original directional bin-centers to the regions
        # found by the voronoi algorithm
        bin_center_regions = []
        for point_region in voro.point_region:
            bin_center_regions.append(voro.regions[point_region])

        return voro.vertices, bin_center_regions

    def direction_num_bins(self):
        return len(self.direction.data)

    def direction_bins_solid_angle(self):
        if not hasattr(self, "_direction_bins_solid_angle"):
            self._direction_bins_solid_angle = (
                self.estimate_direction_bins_solid_angle()
            )
        return self._direction_bins_solid_angle

    def estimate_direction_bins_solid_angle(
        self, seed=43, accuracy=1e-2, max_iterations=1000
    ):
        prng = np.random.Generator(np.random.PCG64(seed))
        num_ii = np.zeros(self.direction_num_bins())
        num_total = 0
        bunch_size = 1000 * 1000

        valid = np.zeros(self.direction_num_bins(), dtype=np.int64)

        itr = 0
        while True:
            itr += 1
            # make random points on hemisphere (positive z-axis)
            points = prng.uniform(low=-1, high=1, size=(bunch_size, 3))

            points[:, 2] = np.abs(points[:, 2])
            norms = np.linalg.norm(points, axis=1)
            points[:, 0] /= norms
            points[:, 1] /= norms
            points[:, 2] /= norms

            dd, ii = self.direction.query(points)
            for i in ii:
                num_ii[i] += 1

            num_total += len(points)
            num_ii_au = np.sqrt(num_ii)
            valid = num_ii > np.sqrt(num_total)

            num_ii_ru = num_ii_au / num_ii
            print(
                "{: 6d}: num. bins valid {: 3d}, ".format(itr, np.sum(valid)),
                "rel. unc.: min: {: .6e}, med: {: .6e}, max: {: .6e}.".format(
                    np.min(num_ii_ru[valid]),
                    np.median(num_ii_ru[valid]),
                    np.max(num_ii_ru[valid]),
                ),
            )
            if np.max(num_ii_ru[valid]) < accuracy:
                break

            if itr > max_iterations:
                raise RuntimeError("Too many iterations")

        hemisphere_solid_angle = 2 * np.pi
        return (
            hemisphere_solid_angle * num_ii / num_total,
            hemisphere_solid_angle * num_ii_ru,
        )

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


def _init_energy_bin_edges(start_GeV, stop_GeV, num_bins):
    return np.geomspace(start_GeV, stop_GeV, num_bins + 1)
