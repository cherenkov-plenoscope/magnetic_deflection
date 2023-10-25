"""
allsky
======
Allsky allows you to populate, store, query the statistics of atmospheric
showers and their deflection due to earth's magnetic field.
"""
import os
import json_utils
import copy
import rename_after_writing as rnw
import atmospheric_cherenkov_response
import binning_utils
import corsika_primary
import svg_cartesian_plot as splt
import numpy as np
import pprint
from . import binning
from . import store
from . import production
from . import dynamicsizerecarray
from .. import corsika
from .. import spherical_coordinates


def init(
    work_dir,
    particle_key="electron",
    site_key="lapalma",
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=8,
    direction_cherenkov_max_zenith_distance_deg=70,
    direction_particle_max_zenith_distance_deg=70,
    direction_num_bins=256,
    population_target_direction_cone_half_angle_deg=3.0,
    population_target_energy_geomspace_factor=1.5,
    population_target_num_showers=10,
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
    assert direction_cherenkov_max_zenith_distance_deg >= 0.0
    assert direction_particle_max_zenith_distance_deg >= 0.0
    assert (
        direction_particle_max_zenith_distance_deg
        <= corsika_primary.MAX_ZENITH_DEG
    )
    assert direction_num_bins >= 1
    assert population_target_direction_cone_half_angle_deg >= 0.0
    assert population_target_energy_geomspace_factor >= 0.0
    assert population_target_num_showers >= 1

    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    site = atmospheric_cherenkov_response.sites.init(site_key)
    with rnw.open(os.path.join(config_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(site, indent=4))

    particle = atmospheric_cherenkov_response.particles.init(particle_key)
    with rnw.open(os.path.join(config_dir, "particle.json"), "wt") as f:
        f.write(json_utils.dumps(particle, indent=4))

    with rnw.open(
        os.path.join(config_dir, "cherenkov_pool_statistics.json"), "wt"
    ) as f:
        f.write(json_utils.dumps({"min_num_cherenkov_photons": 1}, indent=4))

    with rnw.open(
        os.path.join(config_dir, "population_target.json"), "wt"
    ) as f:
        f.write(
            json_utils.dumps(
                {
                    "direction_cone_half_angle_deg": population_target_direction_cone_half_angle_deg,
                    "energy_geomspace_factor": population_target_energy_geomspace_factor,
                    "num_showers": population_target_num_showers,
                },
                indent=4,
            )
        )

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
                    "cherenkov_max_zenith_distance_deg": direction_cherenkov_max_zenith_distance_deg,
                    "particle_max_zenith_distance_deg": direction_particle_max_zenith_distance_deg,
                    "num_bins": direction_num_bins,
                },
                indent=4,
            )
        )

    with rnw.open(os.path.join(config_dir, "corsika_primary.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                {
                    "path": os.path.join(
                        "/",
                        "home",
                        "relleums",
                        "Desktop",
                        "starter_kit",
                        "build",
                        "corsika",
                        "modified",
                        "corsika-75600",
                        "run",
                        "corsika75600Linux_QGSII_urqmd",
                    ),
                },
            )
        )

    config = read_config(work_dir=work_dir)
    assert_config_valid(config=config)

    # storage
    # -------
    store.init(
        store_dir=os.path.join(work_dir, "store"),
        direction_num_bins=direction_num_bins,
        energy_num_bins=energy_num_bins,
    )

    # run_id
    # ------
    production.init(production_dir=os.path.join(work_dir, "production"))


def read_config(work_dir):
    return json_utils.tree.read(os.path.join(work_dir, "config"))


def assert_config_valid(config):
    b = config["binning"]
    assert b["direction"]["cherenkov_max_zenith_distance_deg"] > 0.0
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
    direction_cherenkov_max_zenith_distance_deg=60,
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
        direction_cherenkov_max_zenith_distance_deg
        >= old["binning"]["direction"]["cherenkov_max_zenith_distance_deg"]
    )
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
        direction_cherenkov_max_zenith_distance_deg=direction_cherenkov_max_zenith_distance_deg,
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
        if not _looks_like_a_valid_all_sky_work_dir(work_dir=work_dir):
            raise FileNotFoundError(
                "Does not look like an AllSky() work_dir: '{:s}'.".format(
                    work_dir
                )
            )
        self.work_dir = work_dir
        self.config = read_config(work_dir=work_dir)
        self.binning = binning.Binning(config=self.config["binning"])
        self.store = store.Store(
            store_dir=os.path.join(work_dir, "store"),
            energy_num_bins=self.config["binning"]["energy"]["num_bins"],
            direction_num_bins=self.config["binning"]["direction"]["num_bins"],
        )
        self.production = production.Production(
            production_dir=os.path.join(self.work_dir, "production")
        )

    def _population_make_jobs(self, num_jobs, num_showers_per_job=1000):
        jobs = []
        for j in range(num_jobs):
            job = {}
            job["numer"] = j
            job["work_dir"] = str(self.work_dir)
            job["run_id"] = int(self.production.get_next_run_id_and_bumb())
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

        self.production.lock()

        for ichunk in range(num_chunks):
            jobs = self._population_make_jobs(
                num_jobs=num_jobs,
                num_showers_per_job=num_showers_per_job,
            )
            last_jobs_run_id = jobs[-1]["run_id"]
            results = pool.map(_population_run_job, jobs)
            self.store.commit_stage()
            self.plot_population(
                path=os.path.join(
                    self.work_dir, "run{:06d}.svg".format(last_jobs_run_id)
                )
            )

        self.production.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
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
        max_cer_zd_deg = self.config["binning"]["direction"][
            "cherenkov_max_zenith_distance_deg"
        ]

        vertices, faces = self.binning.direction_delaunay_mesh()
        faces_sol = self.binning.direction_delaunay_mesh_solid_angles()

        cmaps = {}
        for key in ["cherenkov", "particle"]:
            dbin_vertex_values = np.sum(
                self.store._population(key=key), axis=1
            )

            v = np.zeros(len(faces))
            for iface in range(len(faces)):
                face = faces[iface]
                vals = []
                for ee in range(3):
                    if face[ee] < len(dbin_vertex_values):
                        vals.append(dbin_vertex_values[face[ee]])
                v[iface] = np.sum(vals) / len(vals)
                v[iface] /= faces_sol[iface]

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
        splt.shapes.ax_add_circle(
            ax=ax["cherenkov"],
            xy=[0, 0],
            radius=np.sin(np.deg2rad(max_cer_zd_deg)),
            stroke=splt.color.css("blue"),
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

    def query_cherenkov_ball_with_weights(
        self,
        azimuth_deg,
        zenith_deg,
        energy_GeV,
        energy_factor,
        half_angle_deg,
        min_num_cherenkov_photons,
        cut_off_weight=0.05,
    ):
        """
        Parameters
        ----------
        azimuth_deg : float
            Median azimuth angle of Cherenkov-photons in shower.
        zenith_deg : float
            Median zenith angle of Cherenkov-photons in shower.
        energy_GeV : float
            Primary particle's energy.
        energy_factor :
            Query only showers with energies which have energies from:
            energy_GeV*(1-energy_factor) to energy_GeV*(1+energy_factor).
        half_angle_deg : float > 0
            Cone's half angle to query showers in based on the median direction
            of their Cherenkov-photons.
        min_num_cherenkov_photons : float
            Only take showers into account with this many photons.
        cut_off_weight : float
            Ignore showers with weights below this threshold.
        """
        overhead_half_angle_deg = 4.0 * half_angle_deg
        overhead_energy_start_GeV = energy_GeV * (1 - energy_factor) ** 2
        overhead_energy_stop_GeV = energy_GeV * (1 + energy_factor) ** 2

        # print("E", overhead_energy_start_GeV, overhead_energy_stop_GeV)

        dbins, ebins = self.binning.query_ball(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=overhead_half_angle_deg,
            energy_start_GeV=overhead_energy_start_GeV,
            energy_stop_GeV=overhead_energy_stop_GeV,
        )

        debins = []
        for dbin in dbins:
            for ebin in ebins:
                debins.append((dbin, ebin))

        if len(debins) == 0:
            raise RuntimeError("Not enough population")

        # load bins
        # ---------
        if not hasattr(self, "cache"):
            self.cache = {}

        for debin in debins:
            if not debin in self.cache:
                self.cache[debin] = self.store.read(
                    debin[0],
                    debin[1],
                    key="cherenkov",
                )

        colls = []
        dweights = []
        eweights = []
        for debin in debins:
            page = self.cache[debin]

            # direction weights
            # -----------------
            cer_az_deg, cer_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=page["cherenkov_cx_rad"],
                cy=page["cherenkov_cy_rad"],
            )
            arc_deg = spherical_coordinates._angle_between_az_zd_deg(
                az1_deg=azimuth_deg,
                zd1_deg=zenith_deg,
                az2_deg=cer_az_deg,
                zd2_deg=cer_zd_deg,
            )
            arc_weight = gauss1d(x=arc_deg, mean=0.0, sigma=half_angle_deg)
            mask_arc_weight_irrelevant = arc_weight <= cut_off_weight
            # print("arc-cut", np.sum(mask_arc_weight_irrelevant)/len(arc_weight))
            arc_weight[mask_arc_weight_irrelevant] = 0.0

            # energy weights
            # --------------
            energy_weight = gauss1d(
                x=page["particle_energy_GeV"] / energy_GeV,
                mean=1.0,
                sigma=energy_factor,
            )
            mask_energy_weight_irrelevant = energy_weight <= cut_off_weight
            # print("ene-cut", np.sum(mask_energy_weight_irrelevant)/len(energy_weight))
            energy_weight[mask_energy_weight_irrelevant] = 0.0

            weight_mask = np.logical_and(arc_weight > 0.0, energy_weight > 0.0)
            cherenkov_intensity_mask = (
                page["cherenkov_num_photons"] >= min_num_cherenkov_photons
            )

            page_mask = np.logical_and(weight_mask, cherenkov_intensity_mask)
            # print("cherenkov_intensity_mask: ", np.sum(page_mask)/len(page_mask))

            colls.append(page[page_mask])
            dweights.append(arc_weight[page_mask])
            eweights.append(energy_weight[page_mask])

        return np.hstack(colls), np.hstack(dweights), np.hstack(eweights)

    def query_cherenkov_ball(
        self,
        azimuth_deg,
        zenith_deg,
        energy_GeV,
        energy_factor,
        half_angle_deg,
        min_num_cherenkov_photons=1e3,
    ):
        """
        Returns the direction a primary particle must have in order to see its
        Cherenkov-light from a certain direction.

        Parameters
        ----------
        azimuth_deg : float
            Chernekov-light's median azimuth-angle.
        zenith_deg : float
            Cherenkov-light's median zenith-angle.
        energy_GeV : float
            Primary particle's energy.
        energy_factor : float
            Showers with energies in the range (1 - energy_factor) to
            (1 + energy_factor) are taken into account.
        half_angle_deg : float
            Showers within this cone are taken into account.
        """
        colls, dweights, eweights = self.query_cherenkov_ball_with_weights(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            half_angle_deg=half_angle_deg,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )
        weights = dweights * eweights
        sum_weights = np.sum(weights)
        weights /= sum_weights

        if len(weights) == 0 or sum_weights == 0:
            raise RuntimeError("Not enough population.")

        keys = [
            "cherenkov_x_m",
            "cherenkov_y_m",
            "cherenkov_radius50_m",
            "cherenkov_angle50_rad",
            "cherenkov_t_s",
            "cherenkov_t_std_s",
            "cherenkov_num_photons",
            "cherenkov_num_bunches",
        ]

        out = {}
        for key in keys:
            out[key] = weighted_avg_and_std(
                values=colls[key],
                weights=weights,
            )

        par_cx_rad, par_cy_rad = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=colls["particle_azimuth_deg"],
            zenith_deg=colls["particle_zenith_deg"],
        )
        par_cx_rad_avg_std = weighted_avg_and_std(
            values=par_cx_rad,
            weights=weights,
        )
        par_cy_rad_avg_std = weighted_avg_and_std(
            values=par_cy_rad,
            weights=weights,
        )
        (
            par_az_deg_avg,
            par_zd_deg_avg,
        ) = spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=par_cx_rad_avg_std[0],
            cy=par_cy_rad_avg_std[0],
        )
        delta_deg = np.hypot(
            np.rad2deg(par_cx_rad_avg_std[1]),
            np.rad2deg(par_cy_rad_avg_std[1]),
        )

        out["particle_azimuth_deg"] = (par_az_deg_avg, delta_deg)
        out["particle_zenith_deg"] = (par_zd_deg_avg, delta_deg)

        # sample center in population
        # ---------------------------
        cer_cx_rad, cer_cx_rad_std = weighted_avg_and_std(
            values=colls["cherenkov_cx_rad"],
            weights=weights,
        )
        cer_cy_rad, cer_cy_rad_std = weighted_avg_and_std(
            values=colls["cherenkov_cy_rad"],
            weights=weights,
        )
        ene_GeV, ene_GeV_std = weighted_avg_and_std(
            values=colls["particle_energy_GeV"],
            weights=weights,
        )
        cer_az_deg, cer_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=cer_cx_rad,
            cy=cer_cy_rad,
        )

        out["sample"] = {}
        out["sample"]["cherenkov_azimuth_deg"] = cer_az_deg
        out["sample"]["cherenkov_zenith_deg"] = cer_zd_deg
        out["sample"]["energy_GeV"] = ene_GeV
        out["sample"]["weights_num"] = len(weights)
        out["sample"]["weights_sum"] = sum_weights
        return out

    def deflect(
        self,
        cherenkov_directions,
        energy_GeV,
        energy_factor,
        half_angle_deg,
        min_num_cherenkov_photons=1e3,
    ):
        cherenkov_directions = np.array(cherenkov_directions)
        particle_directions = np.zeros(shape=cherenkov_directions.shape)

        for i in range(len(cherenkov_directions)):
            vertex_unitxyz = cherenkov_directions[i]
            vertex_az_deg, vertex_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=vertex_unitxyz[0], cy=vertex_unitxyz[1],
            )

            defl = self.query_cherenkov_ball(
                azimuth_deg=vertex_az_deg,
                zenith_deg=vertex_zd_deg,
                energy_GeV=energy_GeV,
                energy_factor=energy_factor,
                half_angle_deg=half_angle_deg,
                min_num_cherenkov_photons=min_num_cherenkov_photons,
            )

            cx, cy = spherical_coordinates._az_zd_to_cx_cy(
                out["particle_azimuth_deg"], out["particle_zenith_deg"],
            )
            particle_directions[i, 0] = cx
            particle_directions[i, 1] = cy
        return particle_directions


    def plot_deflection(
        self,
        path,
        energy_GeV=5.0,
        energy_factor=0.1,
        num_traces=100,
        min_num_cherenkov_photons=1e3,
    ):
        fig = splt.Fig(cols=1080, rows=1080)
        ax = {}
        ax = splt.hemisphere.Ax(fig=fig)
        ax["span"] = (0.1, 0.1, 0.8, 0.8)

        cer_directions = binning_utils.sphere.fibonacci_space(
            size=num_traces,
            max_zenith_distance_rad=np.deg2rad(60),
        )
        par_directions = []
        for cer_direction in cer_directions:
            cer_az_deg, cer_zd_deg = spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=cer_direction[0],
                cy=cer_direction[1],
            )
            got_it = False

            half_angle_deg = 2
            energy_factor = 0.1
            while not got_it:
                try:
                    # print(half_angle_deg, energy_factor)
                    par_direction = self.query_cherenkov_ball(
                        azimuth_deg=cer_az_deg,
                        zenith_deg=cer_zd_deg,
                        energy_GeV=energy_GeV,
                        energy_factor=energy_factor,
                        half_angle_deg=half_angle_deg,
                        min_num_cherenkov_photons=min_num_cherenkov_photons,
                    )
                    got_it = True
                except RuntimeError as err:
                    half_angle_deg = half_angle_deg**1.1
                    energy_factor = energy_factor**0.9

            # print(par_direction)
            par_directions.append(par_direction)

        for i in range(len(cer_directions)):
            par_direction = par_directions[i]
            if par_direction["sample"]["weights_sum"] * energy_GeV > 100:
                cer_direction = cer_directions[i][0:2]
                par_direction = spherical_coordinates._az_zd_to_cx_cy(
                    azimuth_deg=par_directions[i]["particle_azimuth_deg"][0],
                    zenith_deg=par_directions[i]["particle_zenith_deg"][0],
                )
                cer_direction = np.array(cer_direction)
                par_direction = np.array(par_direction)

                splt.ax_add_line(
                    ax=ax,
                    xy_start=par_direction,
                    xy_stop=0.5 * (par_direction + cer_direction),
                    stroke=splt.color.css("red"),
                )

                splt.ax_add_line(
                    ax=ax,
                    xy_start=0.5 * (par_direction + cer_direction),
                    xy_stop=cer_direction,
                    stroke=splt.color.css("blue"),
                )

        splt.hemisphere.ax_add_grid(ax=ax)

        splt.ax_add_text(
            ax=ax,
            xy=[0.2, 1.15],
            text="primary particle",
            fill=splt.color.css("red"),
            font_family="math",
            font_size=30,
        )
        splt.ax_add_text(
            ax=ax,
            xy=[-0.6, 1.15],
            text="median Cherenkov",
            fill=splt.color.css("blue"),
            font_family="math",
            font_size=30,
        )
        splt.ax_add_text(
            ax=ax,
            xy=[0.3, 1.0],
            text="energy: {: 8.3f}GeV".format(energy_GeV),
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


def _population_run_job(job):
    allsky = AllSky(work_dir=job["work_dir"])

    corsika_steering_dict = production.make_steering(
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

    showers = production.estimate_cherenkov_pool(
        corsika_primary_path=allsky.config["corsika_primary"]["path"],
        corsika_steering_dict=corsika_steering_dict,
        min_num_cherenkov_photons=allsky.config["cherenkov_pool_statistics"][
            "min_num_cherenkov_photons"
        ],
    )
    assert len(showers) == len(corsika_steering_dict["primaries"])

    # staging
    # -------
    cherenkov_stage = allsky.store.make_empty_stage()
    particle_stage = allsky.store.make_empty_stage()

    num_not_enough_light = 0

    for shower in showers:
        if (
            shower["cherenkov_num_photons"]
            >= allsky.config["cherenkov_pool_statistics"][
                "min_num_cherenkov_photons"
            ]
        ):
            # cherenkov
            # ---------
            (
                cer_az_deg,
                cer_zd_deg,
            ) = spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=shower["cherenkov_cx_rad"],
                cy=shower["cherenkov_cy_rad"],
            )

            (delta_phi_deg, delta_energy), (
                dbin,
                ebin,
            ) = allsky.binning.query(
                azimuth_deg=cer_az_deg,
                zenith_deg=cer_zd_deg,
                energy_GeV=shower["particle_energy_GeV"],
            )
            # print("cer", shower["run"], shower["event"], dbin, ebin)
            valid_bin = allsky.binning.is_valid_dbin_ebin(dbin=dbin, ebin=ebin)
            if delta_phi_deg > 10.0 or not valid_bin:
                msg = ""
                msg += "Warning: Shower ({:d},{:d}) is ".format(
                    shower["run"], shower["event"]
                )
                msg += "{:f}deg off the closest cherenkov-bin-center".format(
                    delta_phi_deg
                )
                print(msg)
            else:
                cherenkov_stage["showers"][dbin][ebin].append(
                    copy.deepcopy(shower)
                )
        else:
            num_not_enough_light += 1

        # prticle
        # -------
        (delta_phi_deg, delta_energy), (dbin, ebin) = allsky.binning.query(
            azimuth_deg=shower["particle_azimuth_deg"],
            zenith_deg=shower["particle_zenith_deg"],
            energy_GeV=shower["particle_energy_GeV"],
        )
        # print("par", shower["run"], shower["event"], dbin, ebin)
        valid_bin = allsky.binning.is_valid_dbin_ebin(dbin=dbin, ebin=ebin)
        if delta_phi_deg > 10.0 or not valid_bin:
            msg = ""
            msg += "Warning: Shower ({:d},{:d}) is ".format(
                shower["run"], shower["event"]
            )
            msg += "{:f}deg off the closest particle-bin-center".format(
                delta_phi_deg
            )
            print(msg)
        else:
            particle_stage["showers"][dbin][ebin].append(copy.deepcopy(shower))

    # add to stage
    # ------------
    allsky.store.add_cherenkov_to_stage(cherenkov_stage=cherenkov_stage)
    allsky.store.add_particle_to_stage(particle_stage=particle_stage)

    return True


def _relative_weigth(x, xp, xmin, xmax):
    assert xmin < xp < xmax


def gauss1d(x, mean, sigma):
    return np.exp((-1 / 2) * ((x - mean) ** 2) / (sigma**2))


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def _looks_like_a_valid_all_sky_work_dir(work_dir):
    for dirname in ["config", "store", "production"]:
        if not os.path.isdir(os.path.join(work_dir, dirname)):
            return False
    return True