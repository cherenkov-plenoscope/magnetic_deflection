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
import svg_cartesian_plot as svgplt
import numpy as np
import spherical_coordinates

from . import testing
from . import random
from . import binning
from . import production
from . import store
from . import viewcone
from .. import cherenkov_pool
from .. import utils
from . import hemisphere
from . import analysis
from ..version import __version__


def init(
    work_dir,
    particle_key="electron",
    site_key="lapalma",
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=3,
    direction_particle_max_zenith_distance_rad=None,
    direction_num_bins=8,
):
    """
    Init a new allsky

    Parameters
    ----------
    path : str
        Directory to store the allsky.
    """
    if direction_particle_max_zenith_distance_rad is None:
        direction_particle_max_zenith_distance_rad = (
            corsika_primary.MAX_ZENITH_DISTANCE_RAD
        )
    assert energy_start_GeV > 0.0
    assert energy_stop_GeV > 0.0
    assert energy_stop_GeV > energy_start_GeV
    assert energy_num_bins >= 1
    assert direction_particle_max_zenith_distance_rad >= 0.0
    assert (
        direction_particle_max_zenith_distance_rad
        <= corsika_primary.MAX_ZENITH_DISTANCE_RAD
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
                    "particle_max_zenith_distance_rad": direction_particle_max_zenith_distance_rad,
                    "num_bins": direction_num_bins,
                },
                indent=4,
            )
        )

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
    assert b["direction"]["particle_max_zenith_distance_rad"] > 0.0

    assert b["energy"]["start_GeV"] > 0.0
    assert b["energy"]["stop_GeV"] > 0.0
    assert b["energy"]["num_bins"] > 0
    assert b["energy"]["stop_GeV"] > b["energy"]["start_GeV"]


class AllSky:
    def __init__(self, work_dir, cache_dtype=store.minimal_cache_dtype()):
        """
        Parameters
        ----------
        work_dir : str
            Path to the AllSky's working directory.
        """
        # sniff
        # -----
        if not _looks_like_a_valid_allsky_work_dir(work_dir=work_dir):
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
            cache_dtype=cache_dtype,
        )

    def num_showers(self):
        """
        Returns the number of thrown showers.
        """
        return np.sum(self.store.population_particle())

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
            self.store.commit_stage(pool=pool)

        production_lock.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}', version='{:s}')".format(
            self.__class__.__name__, self.work_dir, self.version
        )
        return out

    def plot_population(self, path):
        fig = svgplt.Fig(cols=1920, rows=1080)
        ax = {}
        ax["cherenkov"] = svgplt.hemisphere.Ax(fig=fig)
        ax["cherenkov"]["span"] = (0.05, 0.1, 0.45, 0.8)

        ax["particle"] = svgplt.hemisphere.Ax(fig=fig)
        ax["particle"]["span"] = (0.55, 0.1, 0.45, 0.8)

        ax["particle_cmap"] = svgplt.hemisphere.Ax(fig=fig)
        ax["particle_cmap"]["span"] = (0.55, 0.05, 0.45, 0.02)

        max_par_zd_rad = self.config["binning"]["direction"][
            "particle_max_zenith_distance_rad"
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
            cmaps[key] = svgplt.color.Map("viridis", start=vmin, stop=vmax)

            mesh_look = svgplt.hemisphere.init_mesh_look(
                num_faces=len(faces),
                stroke=None,
                fill=svgplt.color.css("RoyalBlue"),
                fill_opacity=1.0,
            )

            for i in range(len(faces)):
                mesh_look["faces_fill"][i] = cmaps[key](v[i])

            svgplt.hemisphere.ax_add_mesh(
                ax=ax[key],
                vertices=vertices,
                faces=faces,
                max_radius=1.0,
                **mesh_look,
            )

            svgplt.color.ax_add_colormap(
                ax=ax["particle_cmap"],
                colormap=cmaps[key],
                fn=64,
            )

        svgplt.shapes.ax_add_circle(
            ax=ax["particle"],
            xy=[0, 0],
            radius=np.sin(max_par_zd_rad),
            stroke=svgplt.color.css("red"),
        )
        svgplt.hemisphere.ax_add_grid(ax=ax["particle"])
        svgplt.hemisphere.ax_add_grid(ax=ax["cherenkov"])
        svgplt.ax_add_text(
            ax=ax["cherenkov"],
            xy=[0.0, 1.1],
            text="Cherenkov",
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=30,
        )
        svgplt.ax_add_text(
            ax=ax["particle"],
            xy=[0.0, 1.1],
            text="Particle",
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=30,
        )

        svgplt.fig_write(fig=fig, path=path)

    def plot_query_cherenkov_ball(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_GeV,
        energy_factor,
        path,
        particle_marker_opacity=0.5,
        particle_marker_half_angle_rad=1.0,
        max_num_showers=10000,
        random_shuffle_seed=42,
        min_num_cherenkov_photons=1e3,
    ):
        # query
        _ll = self.query_cherenkov_ball(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            half_angle_rad=half_angle_rad,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )

        # shuffle
        prng = np.random.Generator(np.random.PCG64(random_shuffle_seed))
        shuffled_indices = np.arange(len(_ll))
        prng.shuffle(shuffled_indices)
        ll = _ll[shuffled_indices]

        # plot
        fig = svgplt.Fig(cols=1080, rows=1080)
        ax = {}
        ax = svgplt.hemisphere.Ax(fig=fig)
        ax["span"] = (0.1, 0.1, 0.8, 0.8)

        fov_ring_verts_uxyz = viewcone.make_ring(
            half_angle_rad=half_angle_rad,
            endpoint=True,
            fn=137,
        )
        fov_ring_verts_uxyz = viewcone.rotate(
            vertices_uxyz=fov_ring_verts_uxyz,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            mount="cable_robot_mount",
        )
        par_ring_verts_uxyz = viewcone.make_ring(
            half_angle_rad=particle_marker_half_angle_rad,
            endpoint=False,
            fn=3,
        )

        cmap = svgplt.color.Map(
            "coolwarm",
            start=np.log10(energy_GeV * (1 - energy_factor)),
            stop=np.log10(energy_GeV * (1 + energy_factor)),
        )

        for i in range(min([len(ll), max_num_showers])):
            (
                particle_azimuth_rad,
                particle_zenith_rad,
            ) = spherical_coordinates.cx_cy_to_az_zd(
                ll[i]["particle_cx_rad"],
                ll[i]["particle_cy_rad"],
            )

            rot_par_ring_verts_uxyz = viewcone.rotate(
                vertices_uxyz=par_ring_verts_uxyz,
                azimuth_rad=particle_azimuth_rad,
                zenith_rad=particle_zenith_rad,
                mount="altitude_azimuth_mount",
            )

            svgplt.ax_add_path(
                ax=ax,
                xy=rot_par_ring_verts_uxyz[:, 0:2],
                stroke=None,
                fill=cmap.eval(np.log10(ll[i]["particle_energy_GeV"])),
                fill_opacity=particle_marker_opacity,
            )

        svgplt.ax_add_path(
            ax=ax,
            xy=fov_ring_verts_uxyz[:, 0:2],
            stroke=svgplt.color.css("blue"),
            fill=None,
        )
        svgplt.hemisphere.ax_add_grid(ax=ax)

        svgplt.ax_add_text(
            ax=ax,
            xy=[-0.6, 1.15],
            text="Cherenkov field-of-view",
            fill=svgplt.color.css("blue"),
            font_family="math",
            font_size=30,
        )
        svgplt.ax_add_text(
            ax=ax,
            xy=[0.3, 1.0],
            text="energy: {: 8.3f} to {: 8.3}GeV".format(
                energy_GeV * (1 - energy_factor),
                energy_GeV * (1 + energy_factor),
            ),
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=30,
        )
        svgplt.ax_add_text(
            ax=ax,
            xy=[0.1, -1.05],
            text="site: {:s}".format(self.config["site"]["comment"]),
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=15,
        )
        svgplt.ax_add_text(
            ax=ax,
            xy=[0.1, -1.1],
            text="particle: {:s}".format(self.config["particle"]["key"]),
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=15,
        )

        svgplt.fig_write(fig=fig, path=path)

    def query_cherenkov_ball(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_GeV,
        energy_factor,
        min_num_cherenkov_photons=1e3,
        weights=False,
    ):
        """
        Parameters
        ----------
        azimuth_rad : float
            Median azimuth angle of Cherenkov-photons in shower.
        zenith_rad : float
            Median zenith angle of Cherenkov-photons in shower.
        half_angle_rad : float > 0
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
        overhead_half_angle_rad = 4.0 * half_angle_rad
        overhead_energy_start_GeV = energy_GeV * (1 - energy_factor) ** 2
        overhead_energy_stop_GeV = energy_GeV * (1 + energy_factor) ** 2

        cx, cy = spherical_coordinates.az_zd_to_cx_cy(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
        )

        dir_ene_bins = self.binning.query_ball(
            cx=cx,
            cy=cy,
            half_angle_rad=overhead_half_angle_rad,
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
            cer_az_rad, cer_zd_rad = spherical_coordinates.cx_cy_to_az_zd(
                cx=cer_bin["cherenkov_cx_rad"],
                cy=cer_bin["cherenkov_cy_rad"],
            )
            offaxis_angle_rad = spherical_coordinates.angle_between_az_zd(
                azimuth1_rad=azimuth_rad,
                zenith1_rad=zenith_rad,
                azimuth2_rad=cer_az_rad,
                zenith2_rad=cer_zd_rad,
            )
            viewcone_mask = offaxis_angle_rad <= half_angle_rad

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
                delta_rad = spherical_coordinates.angle_between_cx_cy(
                    cx1=cx,
                    cy1=cy,
                    cx2=matches["cherenkov_cx_rad"][i],
                    cy2=matches["cherenkov_cy_rad"][i],
                )
                dir_weight = utils.gauss1d(
                    x=delta_rad,
                    mean=0.0,
                    sigma=1 / 2 * half_angle_rad,
                )
                ene_weight = utils.gauss1d(
                    x=matches["particle_energy_GeV"][i],
                    mean=energy_GeV,
                    sigma=(energy_GeV * energy_factor),
                )
                direction_weights.append(dir_weight)
                energy_weights.append(ene_weight)

            return (
                matches,
                np.array(direction_weights),
                np.array(energy_weights),
            )


def _population_run_job(job):
    allsky = AllSky(work_dir=job["work_dir"])

    corsika_steering_dict = cherenkov_pool.production.make_steering(
        run_id=job["run_id"],
        site=allsky.config["site"],
        particle_id=allsky.config["particle"]["corsika_particle_id"],
        particle_energy_start_GeV=allsky.binning.energy["start"],
        particle_energy_stop_GeV=allsky.binning.energy["stop"],
        particle_energy_power_slope=-2.0,
        particle_cone_azimuth_rad=0.0,
        particle_cone_zenith_rad=0.0,
        particle_cone_opening_angle_rad=allsky.config["binning"]["direction"][
            "particle_max_zenith_distance_rad"
        ],
        num_showers=job["num_showers_per_job"],
    )

    showers = cherenkov_pool.production.estimate_cherenkov_pool(
        corsika_steering_dict=corsika_steering_dict,
    )
    assert len(showers) == len(corsika_steering_dict["primaries"])

    # staging
    # -------
    cherenkov_stage = allsky.store.make_empty_stage(run_id=job["run_id"])
    particle_stage = allsky.store.make_empty_stage(run_id=job["run_id"])

    for shower in showers:
        if _shower_has_cherenkov_light(shower):
            # cherenkov
            # ---------
            (
                (delta_phi_rad, delta_energy),
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
            (delta_phi_rad, delta_energy),
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


def _shower_has_cherenkov_light(shower):
    return not np.isnan(shower["cherenkov_cx_rad"])


def _looks_like_a_valid_allsky_work_dir(work_dir):
    for dirname in ["config", "store", "production"]:
        if not os.path.isdir(os.path.join(work_dir, dirname)):
            return False
    return True
