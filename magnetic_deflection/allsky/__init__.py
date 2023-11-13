"""
allsky
======
Allsky allows you to populate, store, query the statistics of atmospheric
showers and their deflection due to earth's magnetic field.
"""
import builtins
import os
import json_utils
import copy
import rename_after_writing as rnw
import atmospheric_cherenkov_response
import binning_utils
import corsika_primary
import svg_cartesian_plot as splt
import numpy as np

from . import binning
from . import production
from . import store
from . import viewcone
from .. import spherical_coordinates
from .. import cherenkov_pool
from . import hemisphere
from ..version import __version__


def init(
    work_dir,
    particle_key="electron",
    site_key="lapalma",
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=3,
    direction_particle_max_zenith_distance_deg=70,
    direction_num_bins=8,
    corsika_primary_path=None,
):
    """
    Init a new allsky

    Parameters
    ----------
    path : str
        Directory to store the allsky.
    """
    assert energy_start_GeV > 0.0
    assert energy_stop_GeV > 0.0
    assert energy_stop_GeV > energy_start_GeV
    assert energy_num_bins >= 1
    assert direction_particle_max_zenith_distance_deg >= 0.0
    assert (
        direction_particle_max_zenith_distance_deg
        <= corsika_primary.MAX_ZENITH_DEG
    )
    assert direction_num_bins >= 1

    os.makedirs(work_dir, exist_ok=True)

    with rnw.open(os.path.join(work_dir, "version.json"), "wt") as f:
        f.write(
            json_utils.dumps({"magnetic_deflection": __version__}, indent=4)
        )

    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    site = atmospheric_cherenkov_response.sites.init(site_key)
    with rnw.open(os.path.join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(site, indent=4))

    particle = atmospheric_cherenkov_response.particles.init(particle_key)
    with rnw.open(os.path.join(config_dir, "particle.json"), "wt") as f:
        f.write(json_utils.dumps(particle, indent=4))

    binning_dir = os.path.join(config_dir, "binning")
    os.makedirs(binning_dir, exist_ok=True)

    with rnw.open(os.path.join(binning_dir, "energy.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                {
                    "start_GeV": energy_start_GeV,
                    "stop_GeV": energy_stop_GeV,
                    "num_bins": energy_num_bins,
                },
                indent=4,
            )
        )

    with rnw.open(os.path.join(binning_dir, "direction.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                {
                    "particle_max_zenith_distance_deg": direction_particle_max_zenith_distance_deg,
                    "num_bins": direction_num_bins,
                },
                indent=4,
            )
        )

    if corsika_primary_path is None:
        corsika_primary_path = os.path.join(
            "build",
            "corsika",
            "modified",
            "corsika-75600",
            "run",
            "corsika75600Linux_QGSII_urqmd",
        )
    with rnw.open(os.path.join(config_dir, "corsika_primary.json"), "wt") as f:
        f.write(json_utils.dumps({"path": corsika_primary_path}))

    config = read_config(work_dir=work_dir)
    assert_config_valid(config=config)

    # storage
    # -------
    store.init(
        store_dir=os.path.join(work_dir, "store"),
        num_dir_bins=direction_num_bins,
        num_ene_bins=energy_num_bins,
    )

    # run_id
    # ------
    production.init(production_dir=os.path.join(work_dir, "production"))


def read_config(work_dir):
    return json_utils.tree.read(os.path.join(work_dir, "config"))


def assert_config_valid(config):
    b = config["binning"]
    assert b["direction"]["particle_max_zenith_distance_deg"] > 0.0

    assert b["energy"]["start_GeV"] > 0.0
    assert b["energy"]["stop_GeV"] > 0.0
    assert b["energy"]["num_bins"] > 0
    assert b["energy"]["stop_GeV"] > b["energy"]["start_GeV"]


def rebin(
    inpath,
    outpath,
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=8,
    direction_particle_max_zenith_distance_deg=75,
    direction_num_bins=256,
):
    """
    Read an allsky from inpath and export it to outpath with a different
    binning.
    """
    old = read_config(work_dir=inpath)

    assert energy_start_GeV <= old["binning"]["energy"]["start_GeV"]
    assert energy_stop_GeV >= old["binning"]["energy"]["stop_GeV"]

    assert (
        direction_particle_max_zenith_distance_deg
        >= old["binning"]["direction"]["particle_max_zenith_distance_deg"]
    )

    init(
        work_dir=outpath,
        particle_key=old["particle"]["key"],
        site_key=old["site"]["key"],
        energy_start_GeV=energy_start_GeV,
        energy_stop_GeV=energy_stop_GeV,
        energy_num_bins=energy_num_bins,
        direction_particle_max_zenith_distance_deg=direction_particle_max_zenith_distance_deg,
        direction_num_bins=direction_num_bins,
    )

    raise NotImplementedError("to be done...")


def open(work_dir):
    """
    Open an AllSky.

    Parameters
    ----------
    work_dir : str
        Path to the AllSky's working directory.
    """
    return AllSky(work_dir=work_dir)


class AllSky:
    def __init__(self, work_dir):
        """
        Parameters
        ----------
        work_dir : str
            Path to the AllSky's working directory.
        """
        # sniff
        # -----
        if not _looks_like_a_valid_all_sky_work_dir(work_dir=work_dir):
            raise FileNotFoundError(
                "Does not look like an AllSky() work_dir: '{:s}'.".format(
                    work_dir
                )
            )
        self.work_dir = work_dir

        # version
        # -------
        self.version = self.version_of_when_work_dir_was_initiated()
        if self.version != __version__:
            print(
                (
                    "Warning: The AllSky in '{:s}' ".format(self.work_dir),
                    "was initiated with version {:s}, ".format(self.version),
                    "but this is version {:s}.".format(__version__),
                )
            )

        # allskys' content
        # ----------------
        self.config = read_config(work_dir=work_dir)
        self.binning = binning.Binning(config=self.config["binning"])
        self.store = store.Store(
            store_dir=os.path.join(work_dir, "store"),
        )

    def version_of_when_work_dir_was_initiated(self):
        version_path = os.path.join(self.work_dir, "version.json")
        with builtins.open(version_path, "rt") as f:
            vers = json_utils.loads(f.read())
        return vers["magnetic_deflection"]

    def _populate_make_jobs(
        self, num_jobs, num_showers_per_job, production_lock
    ):
        jobs = []
        for j in range(num_jobs):
            job = {}
            job["numer"] = j
            job["work_dir"] = str(self.work_dir)
            job["run_id"] = int(production_lock.get_next_run_id_and_bumb())
            job["num_showers_per_job"] = int(num_showers_per_job)
            jobs.append(job)
        return jobs

    def populate(
        self, pool, num_chunks=1, num_jobs=1, num_showers_per_job=1000
    ):
        assert num_showers_per_job > 0
        assert num_jobs > 0
        assert num_chunks > 0
        assert os.path.exists(self.config["corsika_primary"]["path"])

        production_lock = production.Production(
            production_dir=os.path.join(self.work_dir, "production")
        )
        production_lock.lock()

        for ichunk in range(num_chunks):
            jobs = self._populate_make_jobs(
                num_jobs=num_jobs,
                num_showers_per_job=num_showers_per_job,
                production_lock=production_lock,
            )
            results = pool.map(_population_run_job, jobs)
            self.store.commit_stage()

        production_lock.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}', version='{:s}')".format(
            self.__class__.__name__, self.work_dir, self.version
        )
        return out

    def plot_population(self, path):
        fig = splt.Fig(cols=1920, rows=1080)
        ax = {}
        ax["cherenkov"] = splt.hemisphere.Ax(fig=fig)
        ax["cherenkov"]["span"] = (0.05, 0.1, 0.45, 0.8)

        ax["particle"] = splt.hemisphere.Ax(fig=fig)
        ax["particle"]["span"] = (0.55, 0.1, 0.45, 0.8)

        ax["particle_cmap"] = splt.hemisphere.Ax(fig=fig)
        ax["particle_cmap"]["span"] = (0.55, 0.05, 0.45, 0.02)

        max_par_zd_deg = self.config["binning"]["direction"][
            "particle_max_zenith_distance_deg"
        ]

        vertices, faces = self.binning.direction_delaunay_mesh()
        faces_sol = self.binning.direction_delaunay_mesh_solid_angles()

        cmaps = {}
        for key in ["cherenkov", "particle"]:
            dbin_vertex_values = np.sum(self.store.population(key=key), axis=1)

            v = np.zeros(len(faces))
            for iface in range(len(faces)):
                face = faces[iface]
                vals = []
                for ee in range(3):
                    if face[ee] < len(dbin_vertex_values):
                        vals.append(dbin_vertex_values[face[ee]])
                v[iface] = np.sum(vals) / len(vals)
                # v[iface] /= faces_sol[iface]

            vmin = 0.0
            vmax = np.max([np.max(v), 1e-6])
            cmaps[key] = splt.color.Map("viridis", start=vmin, stop=vmax)

            mesh_look = splt.hemisphere.init_mesh_look(
                num_faces=len(faces),
                stroke=None,
                fill=splt.color.css("RoyalBlue"),
                fill_opacity=1.0,
            )

            for i in range(len(faces)):
                mesh_look["faces_fill"][i] = cmaps[key](v[i])

            splt.hemisphere.ax_add_mesh(
                ax=ax[key],
                vertices=vertices,
                faces=faces,
                max_radius=1.0,
                **mesh_look,
            )

            splt.color.ax_add_colormap(
                ax=ax["particle_cmap"],
                colormap=cmaps[key],
                fn=64,
            )

        splt.shapes.ax_add_circle(
            ax=ax["particle"],
            xy=[0, 0],
            radius=np.sin(np.deg2rad(max_par_zd_deg)),
            stroke=splt.color.css("red"),
        )
        splt.hemisphere.ax_add_grid(ax=ax["particle"])
        splt.hemisphere.ax_add_grid(ax=ax["cherenkov"])
        splt.ax_add_text(
            ax=ax["cherenkov"],
            xy=[0.0, 1.1],
            text="Cherenkov",
            fill=splt.color.css("black"),
            font_family="math",
            font_size=30,
        )
        splt.ax_add_text(
            ax=ax["particle"],
            xy=[0.0, 1.1],
            text="Particle",
            fill=splt.color.css("black"),
            font_family="math",
            font_size=30,
        )

        splt.fig_write(fig=fig, path=path)

    def plot_query_cherenkov_ball(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_GeV,
        energy_factor,
        path,
        particle_marker_opacity=0.5,
        particle_marker_half_angle_deg=1.0,
        max_num_showers=10000,
        random_shuffle_seed=42,
        min_num_cherenkov_photons=1e3,
    ):
        # query
        _ll = self.query_cherenkov_ball(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            half_angle_deg=half_angle_deg,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )

        # shuffle
        prng = np.random.Generator(np.random.PCG64(random_shuffle_seed))
        shuffled_indices = np.arange(len(_ll))
        prng.shuffle(shuffled_indices)
        ll = _ll[shuffled_indices]

        # plot
        fig = splt.Fig(cols=1080, rows=1080)
        ax = {}
        ax = splt.hemisphere.Ax(fig=fig)
        ax["span"] = (0.1, 0.1, 0.8, 0.8)

        fov_ring_verts_uxyz = viewcone.make_ring(
            half_angle_deg=half_angle_deg,
            endpoint=True,
            fn=137,
        )
        fov_ring_verts_uxyz = viewcone.rotate(
            vertices_uxyz=fov_ring_verts_uxyz,
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            mount="cable_robot_mount",
        )
        par_ring_verts_uxyz = viewcone.make_ring(
            half_angle_deg=particle_marker_half_angle_deg,
            endpoint=False,
            fn=3,
        )

        cmap = splt.color.Map(
            "coolwarm",
            start=np.log10(energy_GeV * (1 - energy_factor)),
            stop=np.log10(energy_GeV * (1 + energy_factor)),
        )

        for i in range(min([len(ll), max_num_showers])):
            (
                particle_azimuth_deg,
                particle_zenith_deg,
            ) = spherical_coordinates._cx_cy_to_az_zd_deg(
                ll[i]["particle_cx_rad"],
                ll[i]["particle_cy_rad"],
            )

            rot_par_ring_verts_uxyz = viewcone.rotate(
                vertices_uxyz=par_ring_verts_uxyz,
                azimuth_deg=particle_azimuth_deg,
                zenith_deg=particle_zenith_deg,
                mount="altitude_azimuth_mount",
            )

            splt.ax_add_path(
                ax=ax,
                xy=rot_par_ring_verts_uxyz[:, 0:2],
                stroke=None,
                fill=cmap.eval(np.log10(ll[i]["particle_energy_GeV"])),
                fill_opacity=particle_marker_opacity,
            )

        splt.ax_add_path(
            ax=ax,
            xy=fov_ring_verts_uxyz[:, 0:2],
            stroke=splt.color.css("blue"),
            fill=None,
        )
        splt.hemisphere.ax_add_grid(ax=ax)

        splt.ax_add_text(
            ax=ax,
            xy=[-0.6, 1.15],
            text="Cherenkov field-of-view",
            fill=splt.color.css("blue"),
            font_family="math",
            font_size=30,
        )
        splt.ax_add_text(
            ax=ax,
            xy=[0.3, 1.0],
            text="energy: {: 8.3f} to {: 8.3}GeV".format(
                energy_GeV * (1 - energy_factor),
                energy_GeV * (1 + energy_factor),
            ),
            fill=splt.color.css("black"),
            font_family="math",
            font_size=30,
        )
        splt.ax_add_text(
            ax=ax,
            xy=[0.1, -1.05],
            text="site: {:s}".format(self.config["site"]["comment"]),
            fill=splt.color.css("black"),
            font_family="math",
            font_size=15,
        )
        splt.ax_add_text(
            ax=ax,
            xy=[0.1, -1.1],
            text="particle: {:s}".format(self.config["particle"]["key"]),
            fill=splt.color.css("black"),
            font_family="math",
            font_size=15,
        )

        splt.fig_write(fig=fig, path=path)

    def query_cherenkov_ball(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_GeV,
        energy_factor,
        min_num_cherenkov_photons=1e3,
        weights=False,
    ):
        """
        Parameters
        ----------
        azimuth_deg : float
            Median azimuth angle of Cherenkov-photons in shower.
        zenith_deg : float
            Median zenith angle of Cherenkov-photons in shower.
        half_angle_deg : float > 0
            Cone's half angle to query showers in based on the median direction
            of their Cherenkov-photons.
        energy_GeV : float
            Primary particle's energy.
        energy_factor :
            Query only showers with energies which have energies from:
            energy_GeV*(1-energy_factor) to energy_GeV*(1+energy_factor).
        min_num_cherenkov_photons : float
            Only take showers into account with this many photons.
        weights : bool
            If true, weights will be added to the output indicating the
            proximity to the queried direction and energy.

        Returns
        -------
        matches : numpy.recarray
            All showers which match the query.
        """
        overhead_half_angle_deg = 4.0 * half_angle_deg
        overhead_energy_start_GeV = energy_GeV * (1 - energy_factor) ** 2
        overhead_energy_stop_GeV = energy_GeV * (1 + energy_factor) ** 2

        cx, cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
        )

        dir_ene_bins = self.binning.query_ball(
            cx=cx,
            cy=cy,
            half_angle_deg=overhead_half_angle_deg,
            energy_start_GeV=overhead_energy_start_GeV,
            energy_stop_GeV=overhead_energy_stop_GeV,
        )

        energy_start_GeV = energy_GeV * (1 - energy_factor)
        energy_stop_GeV = energy_GeV * (1 + energy_factor)

        if len(dir_ene_bins) == 0:
            raise RuntimeError("Not enough population")

        colls = []
        for dir_ene_bin in dir_ene_bins:
            cer_bin = self.store[dir_ene_bin]

            # direction
            # ---------
            cer_az_deg, cer_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=cer_bin["cherenkov_cx_rad"],
                cy=cer_bin["cherenkov_cy_rad"],
            )
            offaxis_angle_deg = spherical_coordinates._angle_between_az_zd_deg(
                az1_deg=azimuth_deg,
                zd1_deg=zenith_deg,
                az2_deg=cer_az_deg,
                zd2_deg=cer_zd_deg,
            )
            viewcone_mask = offaxis_angle_deg <= half_angle_deg

            # energy
            # ------
            energy_mask = np.logical_and(
                cer_bin["particle_energy_GeV"] >= energy_start_GeV,
                cer_bin["particle_energy_GeV"] < energy_stop_GeV,
            )

            # photon statistics
            # -----------------
            cherenkov_intensity_mask = (
                cer_bin["cherenkov_num_photons"] >= min_num_cherenkov_photons
            )

            mask = viewcone_mask * energy_mask * cherenkov_intensity_mask

            colls.append(cer_bin[mask])

        matches = np.hstack(colls)

        if not weights:
            return matches
        else:
            direction_weights = []
            energy_weights = []
            for i in range(len(matches)):
                delta_rad = spherical_coordinates._angle_between_cx_cy_rad(
                    cx1=cx,
                    cy1=cy,
                    cx2=matches["cherenkov_cx_rad"][i],
                    cy2=matches["cherenkov_cy_rad"][i],
                )
                dir_weight = gauss1d(
                    x=delta_rad,
                    mean=0.0,
                    sigma=1 / 2 * half_angle_deg,
                )
                ene_weight = gauss1d(
                    x=matches["particle_energy_GeV"][i],
                    mean=energy_GeV,
                    sigma=(1 / 2 * energy_GeV * energy_factor),
                )
                direction_weights.append(dir_weight)
                energy_weights.append(ene_weight)

            return (
                matches,
                np.array(direction_weights),
                np.array(energy_weights),
            )

    def query_cherenkov_ball_into_grid_mask(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_GeV,
        energy_factor,
        shower_spread_half_angle_deg,
        min_num_cherenkov_photons,
        hemisphere_grid,
    ):
        """
        For a given hemispherical grid, this returns a mask for this grid
        indicating the grid's faces which contain primary particles which will
        lead to the requested Cherenkov-light.

        Parameters
        ----------
        azimuth_deg : float
            Median azimuth angle of Cherenkov-photons in shower.
        zenith_deg : float
            Median zenith angle of Cherenkov-photons in shower.
        half_angle_deg : float > 0
            Cone's half angle to query showers in based on the median direction
            of their Cherenkov-photons.
        energy_GeV : float
            Primary particle's energy.
        energy_factor :
            Query only showers with energies which have energies from:
            energy_GeV*(1-energy_factor) to energy_GeV*(1+energy_factor).
        shower_spread_half_angle_deg : float >= 0
            The mask will be dilluted by this angle to account for spread in
            the shower's development.
        min_num_cherenkov_photons : float
            Only take showers into account with this many photons.
        hemisphere_grid : hemisphere.Grid
            The geometry of the grid defining its faces (tiles).

        Returns
        -------
        hemisphere_mask : hemisphere.Mask
            A mask marking the grid's faces which are possible candidates to
            throw primary particles froms.
        """
        matches = self.query_cherenkov_ball(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=half_angle_deg,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )

        hemisphere_mask = hemisphere.Mask(mesh=hemisphere_grid)
        for match in matches:
            hemisphere_mask.append_cx_cy(
                cx=match["particle_cx_rad"],
                cy=match["particle_cy_rad"],
                half_angle_deg=shower_spread_half_angle_deg,
            )
        return hemisphere_mask


def gauss1d(x, mean, sigma):
    return np.exp((-1 / 2) * ((x - mean) ** 2) / (sigma**2))


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- numpy arrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def _population_run_job(job):
    allsky = AllSky(work_dir=job["work_dir"])

    corsika_steering_dict = cherenkov_pool.production.make_steering(
        run_id=job["run_id"],
        site=allsky.config["site"],
        particle_id=allsky.config["particle"]["corsika_particle_id"],
        particle_energy_start_GeV=allsky.binning.energy["start"],
        particle_energy_stop_GeV=allsky.binning.energy["stop"],
        particle_energy_power_slope=-2.0,
        particle_cone_azimuth_deg=0.0,
        particle_cone_zenith_deg=0.0,
        particle_cone_opening_angle_deg=allsky.config["binning"]["direction"][
            "particle_max_zenith_distance_deg"
        ],
        num_showers=job["num_showers_per_job"],
    )

    showers = cherenkov_pool.production.estimate_cherenkov_pool(
        corsika_primary_path=allsky.config["corsika_primary"]["path"],
        corsika_steering_dict=corsika_steering_dict,
    )
    assert len(showers) == len(corsika_steering_dict["primaries"])

    # staging
    # -------
    cherenkov_stage = allsky.store.make_empty_stage(run_id=job["run_id"])
    particle_stage = allsky.store.make_empty_stage(run_id=job["run_id"])

    for shower in showers:
        if shower_has_cherenkov_light(shower):
            # cherenkov
            # ---------
            (
                (delta_phi_deg, delta_energy),
                dir_ene_bin,
            ) = allsky.binning.query(
                cx=shower["cherenkov_cx_rad"],
                cy=shower["cherenkov_cy_rad"],
                energy_GeV=shower["particle_energy_GeV"],
            )

            valid_bin = allsky.binning.is_valid_dir_ene_bin(
                dir_ene_bin=dir_ene_bin
            )

            cherenkov_stage["records"][dir_ene_bin].append(
                copy.deepcopy(shower)
            )

        # prticle
        # -------
        (
            (delta_phi_deg, delta_energy),
            dir_ene_bin,
        ) = allsky.binning.query(
            cx=shower["particle_cx_rad"],
            cy=shower["particle_cy_rad"],
            energy_GeV=shower["particle_energy_GeV"],
        )

        valid_bin = allsky.binning.is_valid_dir_ene_bin(
            dir_ene_bin=dir_ene_bin
        )

        particle_stage["records"][dir_ene_bin].append(copy.deepcopy(shower))

    # add to stage
    # ------------
    allsky.store.add_cherenkov_to_stage(cherenkov_stage=cherenkov_stage)
    allsky.store.add_particle_to_stage(particle_stage=particle_stage)

    return True


def shower_has_cherenkov_light(shower):
    return not np.isnan(shower["cherenkov_cx_rad"])


def _looks_like_a_valid_all_sky_work_dir(work_dir):
    for dirname in ["config", "store", "production"]:
        if not os.path.isdir(os.path.join(work_dir, dirname)):
            return False
    return True


def draw_particle_direction_with_cone(
    prng,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    energy_GeV,
    energy_factor,
    shower_spread_half_angle_deg,
    min_num_cherenkov_photons,
    allsky_deflection,
    max_iterations=1000 * 1000,
):
    (
        matches,
        direction_weights,
        energy_weights,
    ) = allsky_deflection.query_cherenkov_ball(
        azimuth_deg=azimuth_deg,
        zenith_deg=zenith_deg,
        half_angle_deg=half_angle_deg,
        energy_GeV=energy_GeV,
        energy_factor=energy_factor,
        hemisphere_grid=hemisphere_grid,
        shower_spread_half_angle_deg=shower_spread_half_angle_deg,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
        weights=True,
    )
    avg_particle_cx_rad, std_particle_cx_rad = weighted_avg_and_std(
        values=matches["particle_cx_rad"], weights=energy_weights
    )
    avg_particle_cy_rad, std_particle_cy_rad = weighted_avg_and_std(
        values=matches["particle_cy_rad"], weights=energy_weights
    )

    (
        avg_particle_azimuth_deg,
        avg_particle_zenith_deg,
    ) = spherical_coordinates._cx_cy_to_az_zd_deg(
        cx=avg_particle_cx_rad, cy=avg_particle_cy_rad
    )

    half_angle_thrown_rad = np.deg2rad(
        shower_spread_half_angle_deg
    ) + np.deg2rad(half_angle_deg)
    (
        particle_azimuth_rad,
        particle_zenith_rad,
    ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
        prng=prng,
        azimuth_rad=np.deg2rad(avg_particle_azimuth_deg),
        zenith_rad=np.deg2rad(avg_particle_zenith_deg),
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=half_angle_thrown_rad,
        max_zenith_rad=np.deg2rad(corsika_primary.MAX_ZENITH_DEG),
        max_iterations=max_iterations,
    )
    return {
        "particle_azimuth_rad": particle_azimuth_rad,
        "particle_zenith_rad": particle_zenith_rad,
        "solid_angle_thrown_sr": solid_angle_utils.cone.solid_angle(
            half_angle_rad=half_angle_thrown_rad
        ),
    }


def draw_particle_direction_with_masked_grid(
    prng,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    energy_GeV,
    energy_factor,
    shower_spread_half_angle_deg,
    min_num_cherenkov_photons,
    allsky_deflection,
    hemisphere_grid,
    max_iterations=1000 * 1000,
):
    """
    Parameters
    ----------
    prng : numpy.random.Generator
        The pseudo random number-generator to draw from.

    Returns
    -------
        : dict

        particle_azimuth_rad : float

        particle_zenith_rad : float

        solid_angle_thrown_sr : float
            The total solid angle of all masked faces in the hemispherical grid
            which where thrown in.
    """

    hemisphere_mask = allsky_deflection.query_cherenkov_ball_into_grid_mask(
        azimuth_deg=azimuth_deg,
        zenith_deg=zenith_deg,
        half_angle_deg=half_angle_deg,
        energy_GeV=energy_GeV,
        energy_factor=energy_factor,
        hemisphere_grid=hemisphere_grid,
        shower_spread_half_angle_deg=shower_spread_half_angle_deg,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
    )

    iteration = 0
    hit = False
    while not hit:
        (
            particle_azimuth_rad,
            particle_zenith_rad,
        ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=np.deg2rad(
                corsika_primary.MAX_ZENITH_DEG
            ),
            max_zenith_rad=np.deg2rad(corsika_primary.MAX_ZENITH_DEG),
            max_iterations=max_iterations,
        )

        face_idx = hemisphere_grid.query_azimuth_zenith(
            azimuth_deg=np.rad2deg(particle_azimuth_rad),
            zenith_deg=np.rad2deg(particle_zenith_rad),
        )

        if face_idx in hemisphere_mask.faces:
            hit = True

        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Rejection-sampling failed.")

    return {
        "particle_azimuth_rad": particle_azimuth_rad,
        "particle_zenith_rad": particle_zenith_rad,
        "solid_angle_thrown_sr": hemisphere_mask.solid_angle(),
    }
