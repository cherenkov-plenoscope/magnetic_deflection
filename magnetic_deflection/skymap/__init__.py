from .. import utils
from ..version import __version__
from .. import allsky
from .. import cherenkov_pool
from . import recarray_utils
from . import plotting
from . import querying

import io
import os
import numpy as np
import copy
import gzip
import glob
import tarfile

import atmospheric_cherenkov_response
import rename_after_writing as rnw
import json_utils
import solid_angle_utils
import spherical_histogram
import spherical_coordinates
import binning_utils
import corsika_primary
import triangle_mesh_io
import dynamicsizerecarray
import svg_cartesian_plot as svgplt


def init(
    work_dir,
    particle_key,
    site_key,
    energy_bin_edges_GeV,
    energy_power_slope,
    threshold_num_photons,
    sky_vertices,
    sky_faces,
    ground_bin_area_m2,
):
    opj = os.path.join

    os.makedirs(work_dir, exist_ok=True)
    _write_versions(
        path=opj(work_dir, "versions.json"),
        versions=_gather_versions_now(),
    )

    config_dir = opj(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # site
    # ----
    site = atmospheric_cherenkov_response.sites.init(site_key)
    _json_write(path=opj(config_dir, "site.json"), o=site)

    # particle
    # --------
    particle = atmospheric_cherenkov_response.particles.init(particle_key)
    _json_write(path=opj(config_dir, "particle.json"), o=particle)

    # binning
    # -------
    binning_dir = opj(config_dir, "binning")
    os.makedirs(binning_dir, exist_ok=True)

    # energy
    # ------
    assert binning_utils.is_strictly_monotonic_increasing(energy_bin_edges_GeV)
    _json_write(
        path=opj(binning_dir, "energy_bin_edges_GeV.json"),
        o=energy_bin_edges_GeV,
    )

    assert not np.isnan(energy_power_slope)
    _json_write(
        path=opj(config_dir, "energy_power_slope.json"),
        o=energy_power_slope,
    )

    # sky
    # ---
    sky_obj = spherical_histogram.mesh.vertices_and_faces_to_obj(
        vertices=sky_vertices,
        faces=sky_faces,
        mtlkey="sky",
    )
    with rnw.open(opj(binning_dir, "sky.obj"), "wt") as f:
        f.write(triangle_mesh_io.obj.dumps(sky_obj))

    # ground
    # ------
    assert ground_bin_area_m2 > 0.0
    _json_write(
        path=opj(binning_dir, "ground_bin_area_m2.json"),
        o=ground_bin_area_m2,
    )

    # threshold
    # ---------
    assert threshold_num_photons > 0.0
    _json_write(
        path=opj(binning_dir, "threshold_num_photons.json"),
        o=threshold_num_photons,
    )

    # run_id
    # ------
    allsky.production.init(production_dir=opj(work_dir, "production"))

    return SkyMap(work_dir=work_dir)


def init_example(work_dir, particle_key="electron", site_key="lapalma"):
    PORTAL_FOV_HALF_ANGLE_RAD = np.deg2rad(3.25)
    PORTAL_MIRROR_DIAMETER_M = 71.0
    PORTAL_THRESHOLD_NUM_PHOTONS = 25
    overhead = 2.0

    vertices, faces = _guess_sky_vertices_and_faces(
        fov_half_angle_rad=PORTAL_FOV_HALF_ANGLE_RAD,
        num_faces_in_fov=overhead,
        max_zenith_distance_rad=np.deg2rad(89.0),
    )
    ground_bin_area_m2 = _guess_ground_bin_area_m2(
        mirror_diameter_m=PORTAL_MIRROR_DIAMETER_M,
        num_bins_in_mirror=overhead,
    )

    ENERGY_POWER_SLOPE = -1.5

    particle = atmospheric_cherenkov_response.particles.init(particle_key)
    energy_start_GeV = atmospheric_cherenkov_response.particles.compile_energy(
        particle["population"]["energy"]["start_GeV"]
    )

    return init(
        work_dir=work_dir,
        particle_key=particle_key,
        site_key=site_key,
        energy_bin_edges_GeV=_guess_energy_bin_edges_GeV(
            energy_power_slope=ENERGY_POWER_SLOPE,
            energy_start_GeV=energy_start_GeV,
        ),
        energy_power_slope=ENERGY_POWER_SLOPE,
        threshold_num_photons=(PORTAL_THRESHOLD_NUM_PHOTONS / overhead),
        sky_vertices=vertices,
        sky_faces=faces,
        ground_bin_area_m2=ground_bin_area_m2,
    )


class SkyMap:
    def __init__(self, work_dir):
        opj = os.path.join

        self.work_dir = os.path.abspath(work_dir)

        self.versions = {
            "work_dir": _read_versions(path=opj(work_dir, "versions.json")),
            "instance": _gather_versions_now(),
        }
        if not _versions_equal(
            self.versions["instance"], self.versions["work_dir"]
        ):
            print("SkyMap {:s} has a different version.".format(self.work_dir))

        # read
        # ----
        self.config = json_utils.tree.read(path=opj(work_dir, "config"))
        cfg = self.config
        self.threshold_num_photons = cfg["binning"]["threshold_num_photons"]

        with open(opj(work_dir, "config", "binning", "sky.obj"), "rt") as f:
            _sky_obj = triangle_mesh_io.obj.loads(f.read())

        self.binning = {}
        self.binning["energy"] = binning_utils.Binning(
            bin_edges=cfg["binning"]["energy_bin_edges_GeV"]
        )

        (
            _sky_vertices,
            _sky_faces,
        ) = spherical_histogram.mesh.obj_to_vertices_and_faces(obj=_sky_obj)
        self.binning["sky"] = spherical_histogram.geometry.HemisphereGeometry(
            vertices=_sky_vertices,
            faces=_sky_faces,
        )
        self.binning["ground"] = {}
        self.binning["ground"]["area"] = cfg["binning"]["ground_bin_area_m2"]
        self.binning["ground"]["width"] = np.sqrt(
            self.binning["ground"]["area"]
        )

    def _populate_make_jobs(
        self, num_jobs, num_showers_per_job, production_lock
    ):
        jobs = []
        for j in range(num_jobs):
            job = {}
            job["work_dir"] = str(self.work_dir)
            job["run_id"] = int(production_lock.get_next_run_id_and_bumb())
            job["num_showers"] = int(num_showers_per_job)
            jobs.append(job)
        return jobs

    def populate(
        self, pool, num_chunks=1, num_jobs=1, num_showers_per_job=1000
    ):
        assert num_showers_per_job > 0
        assert num_jobs > 0
        assert num_chunks > 0

        production_lock = allsky.production.Production(
            production_dir=os.path.join(self.work_dir, "production")
        )
        production_lock.lock()

        for ichunk in range(num_chunks):
            jobs = self._populate_make_jobs(
                num_jobs=num_jobs,
                num_showers_per_job=num_showers_per_job,
                production_lock=production_lock,
            )
            _ = pool.map(_population_run_job, jobs)
            _collect_stage_into_results(work_dir=self.work_dir, skymap=self)

        production_lock.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out

    def _query_ball_bins(
        self,
        query,
    ):
        skybins, skybins_weights = self.binning[
            "sky"
        ].query_cone_weiths_azimuth_zenith(
            azimuth_rad=query["azimuth_rad"],
            zenith_rad=query["zenith_rad"],
            half_angle_rad=query["half_angle_rad"],
        )

        enebins, enebins_absolute_weights = binning_utils.power.query_ball(
            bin_edges=self.binning["energy"]["edges"],
            start=query["energy_start_GeV"],
            stop=query["energy_stop_GeV"],
            power_slope=self.config["energy_power_slope"],
        )
        if len(enebins) > 0:
            enebins_weights = enebins_absolute_weights / np.sum(
                enebins_absolute_weights
            )
        else:
            enebins_weights = np.array([])

        return (skybins, skybins_weights, enebins, enebins_weights)

    def query_ball(
        self,
        query,
        map_key,
    ):
        (
            skybins,
            skybins_weights,
            enebins,
            enebins_weights,
        ) = self._query_ball_bins(query=query)

        fsky, fene, fweights = _combine_weights_of_sky_and_ene_bins(
            skybins=skybins,
            skybins_weights=skybins_weights,
            enebins=enebins,
            enebins_weights=enebins_weights,
        )

        ex = self.map_exposure()
        if map_key == "primary_to_cherenkov":
            mmm = self.map_primary_to_cherenkov()
        elif map_key == "cherenkov_to_primary":
            mmm = self.map_cherenkov_to_primary()
        else:
            raise ValueError("Unknown map_key '{:s}'.".format(map_key))

        sky_intensity = np.zeros(len(self.binning["sky"].faces))

        for i in range(len(fsky)):
            skybin = fsky[i]
            enebin = fene[i]
            weight = fweights[i]
            if ex[enebin][skybin] > 0:
                cell_sky_intensity = mmm[enebin][skybin] / ex[enebin][skybin]
                sky_intensity += weight * cell_sky_intensity

        return sky_intensity

    def plot_query_ball(
        self,
        query,
        map_key,
        path,
        quantile=0.0,
        vmin=None,
        vmax=None,
        logmin=1e-6,
        logscale=True,
    ):
        sky_intensity = self.query_ball(query=query, map_key=map_key)

        if map_key == "primary_to_cherenkov":
            sky_intensity /= self.binning["sky"].faces_solid_angles

        # estimate p50 of p50 brightest faces
        _sky_p50_mask = (
            spherical_histogram.mask_fewest_bins_to_contain_quantile(
                bin_counts=sky_intensity,
                quantile=0.5,
            )
        )
        _sky_brightest_faces = sky_intensity[_sky_p50_mask]
        sky_intensity_p50 = np.quantile(_sky_brightest_faces, q=0.5)

        if quantile is not None:
            _sky_intensity_quantile_mask = (
                spherical_histogram.mask_fewest_bins_to_contain_quantile(
                    bin_counts=sky_intensity,
                    quantile=quantile,
                )
            )
            sky_intensity_quantile_mask = (
                spherical_histogram.mesh.fill_faces_mask_if_two_neighbors_true(
                    faces_mask=_sky_intensity_quantile_mask,
                    faces_neighbors=self.binning["sky"].faces_neighbors,
                )
            )
        else:
            sky_intensity_quantile_mask = None

        if logscale:
            sky_intensity += logmin
            intensity_scale = svgplt.scaling.log(base=10)
        else:
            intensity_scale = svgplt.scaling.unity()

        if vmin is None:
            vmin = min(sky_intensity)
        if vmax is None:
            vmax = max(sky_intensity)

        if map_key == "primary_to_cherenkov":
            colormap = svgplt.color.Map(
                name="inferno",
                start=vmin,
                stop=vmax,
                func=intensity_scale,
            )
            sky_intensity_unit = "cherenkov density / (sr)\u207b\u00b9"

        elif map_key == "cherenkov_to_primary":
            colormap = svgplt.color.Map(
                name="viridis",
                start=vmin,
                stop=vmax,
                func=intensity_scale,
            )
            sky_intensity_unit = "trigger prob. / 1"

        else:
            raise ValueError("No such map_key '{:s}'".format(map_key))

        NPIX = 1280
        fig = svgplt.Fig(cols=(3 * NPIX) // 2, rows=NPIX)
        font_size = 15 * NPIX / 1280
        stroke_width = NPIX / 1280

        ax = svgplt.hemisphere.Ax(fig=fig)
        ax["span"] = (0.0, 0.0, 1 / (3 / 2), 1)

        axw = svgplt.Ax(fig=fig)
        axw["span"] = (0.7, 0.1, 0.025, 0.8)
        axw["yscale"] = colormap.func

        axe = svgplt.Ax(fig=fig)
        axe["span"] = (0.85, 0.1, 0.05, 0.8)

        plotting.ax_add_sky(
            ax=ax,
            sky_vertices=self.binning["sky"].vertices,
            sky_faces=self.binning["sky"].faces,
            sky_intensity=sky_intensity,
            colormap=colormap,
            fill_opacity=1.0,
            sky_mask=sky_intensity_quantile_mask,
            sky_mask_color=svgplt.color.css("orange"),
        )

        plotting.ax_add_fov(
            ax=ax,
            azimuth_rad=query["azimuth_rad"],
            zenith_rad=query["zenith_rad"],
            half_angle_rad=query["half_angle_rad"],
            stroke=svgplt.color.css("red"),
            stroke_width=4 * stroke_width,
            fill=None,
        )

        plotting.ax_add_energy_bar(
            ax=axe,
            bin_edges=self.binning["energy"]["edges"],
            power_slope=self.config["energy_power_slope"],
            start=query["energy_start_GeV"],
            stop=query["energy_stop_GeV"],
            font_size=3 * font_size,
            stroke_width=1.5 * stroke_width,
            stroke=svgplt.color.css("black"),
        )

        svgplt.color.ax_add_colormap(
            ax=axw,
            colormap=colormap,
            fn=128,
            orientation="vertical",
        )
        svgplt.color.ax_add_colormap_ticks(
            ax=axw,
            colormap=colormap,
            num=6,
            orientation="vertical",
            fill=svgplt.color.css("black"),
            stroke=None,
            stroke_width=1.5 * stroke_width,
            font_family="math",
            font_size=3 * font_size,
        )
        svgplt.ax_add_line(
            ax=axw,
            xy_start=[-0.5, sky_intensity_p50],
            xy_stop=[0, sky_intensity_p50],
            stroke=svgplt.color.css("black"),
            stroke_width=5 * stroke_width,
        )

        svgplt.hemisphere.ax_add_grid(
            ax=ax,
            stroke=svgplt.color.css("white"),
            stroke_opacity=1.0,
            stroke_width=0.3 * stroke_width,
            font_size=3.0 * font_size,
        )
        svgplt.ax_add_text(
            ax=axe,
            xy=[-5, -0.075],
            text=sky_intensity_unit,
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=3 * font_size,
        )
        svgplt.ax_add_text(
            ax=axe,
            xy=[0.0, -0.075],
            text="energy / GeV",
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=3 * font_size,
        )
        svgplt.ax_add_text(
            ax=ax,
            xy=[-1.0, -0.85],
            text="{:s}".format(self.config["site"]["key"]),
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=3 * font_size,
        )
        svgplt.ax_add_text(
            ax=ax,
            xy=[-1.0, -1],
            text="{:s}".format(self.config["particle"]["key"]),
            fill=svgplt.color.css("black"),
            font_family="math",
            font_size=3 * font_size,
        )

        svgplt.fig_write(fig=fig, path=path + ".svg")

        from svg_cartesian_plot import inkscape

        inkscape.render(
            path + ".svg",
            path,
            background_opacity=0.0,
            export_type="png",
        )
        os.remove(path + ".svg")

        return sky_intensity

    def demonstrate(
        self,
        path,
        pool,
        num_jobs=6,
        queries=None,
        map_key="primary_to_cherenkov",
    ):
        os.makedirs(path, exist_ok=True)
        if queries is None:
            queries = querying.example(num=6)

        map_keys = {
            "primary_to_cherenkov": {
                "vmin": 1e1,
                "vmax": 1e8,
                "quantile": 0,
                "logscale": True,
            },
            "cherenkov_to_primary": {
                "vmin": 1e-2,
                "vmax": 1.25,
                "quantile": 0.8,
                "logscale": False,
            },
        }

        num_tasks = int(np.ceil(len(queries) / num_jobs) * num_jobs)
        task_ii = np.arange(num_tasks)

        job_ii = np.split(task_ii, num_jobs)

        jobs = []
        for j in range(len(job_ii)):
            job = {}
            job["work_dir"] = self.work_dir
            job["calls"] = []
            for i in job_ii[j]:
                if i < len(queries):
                    name = "{:06d}".format(i)
                    imgpath = os.path.join(path, name + "." + map_key + ".png")
                    call = {}
                    call["query"] = queries[i]
                    call["map_key"] = map_key
                    call["path"] = imgpath
                    call["quantile"] = map_keys[map_key]["quantile"]
                    call["vmin"] = map_keys[map_key]["vmin"]
                    call["vmax"] = map_keys[map_key]["vmax"]
                    call["logscale"] = map_keys[map_key]["logscale"]
                    job["calls"].append(call)
            jobs.append(job)

        pool.map(_run_job_plot_query_ball, jobs)

        try:
            from sebastians_matplotlib_addons import video

            video.write_video_from_image_slices(
                image_sequence_wildcard_path=os.path.join(
                    path, "%06d" + "." + map_key + ".png"
                ),
                output_path=os.path.join(path, map_key + ".mov"),
            )
        except:
            pass

    def map_primary_to_cherenkov(self):
        if not hasattr(self, "_map_primary_to_cherenkov"):
            self._map_primary_to_cherenkov = utils.read_array(
                path=os.path.join(
                    self.work_dir, "results", "map.primary_to_cherenkov.rec"
                )
            )
        return self._map_primary_to_cherenkov

    def map_cherenkov_to_primary(self):
        if not hasattr(self, "_map_cherenkov_to_primary"):
            self._map_cherenkov_to_primary = utils.read_array(
                path=os.path.join(
                    self.work_dir, "results", "map.cherenkov_to_primary.rec"
                )
            )
        return self._map_cherenkov_to_primary

    def map_exposure(self):
        if not hasattr(self, "_map_exposure"):
            self._map_exposure = utils.read_array(
                path=os.path.join(self.work_dir, "results", "map.exposure.rec")
            )
        return self._map_exposure


def _guess_energy_bin_edges_GeV(
    energy_power_slope, energy_start_GeV=2 ** (-2)
):
    return binning_utils.power.space(
        start=energy_start_GeV,
        stop=2 ** (6),
        power_slope=energy_power_slope,
        size=32 + 1,
    )


def _guess_sky_vertices_and_faces(
    fov_half_angle_rad,
    num_faces_in_fov,
    max_zenith_distance_rad=np.deg2rad(89),
):
    sky_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=max_zenith_distance_rad
    )
    fov_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=fov_half_angle_rad
    )
    num_faces = num_faces_in_fov * sky_sr / fov_sr

    APPROX_NUM_VERTICES_PER_FACE = 0.5
    num_vertices = APPROX_NUM_VERTICES_PER_FACE * num_faces

    num_vertices = int(np.ceil(num_vertices))

    vertices = spherical_histogram.mesh.make_vertices(
        num_vertices=num_vertices,
        max_zenith_distance_rad=max_zenith_distance_rad,
    )
    faces = spherical_histogram.mesh.make_faces(vertices=vertices)
    return vertices, faces


def _guess_ground_bin_area_m2(mirror_diameter_m, num_bins_in_mirror):
    mirror_radius_m = 0.5 * mirror_diameter_m
    mirror_area_m2 = np.pi * mirror_radius_m**2
    ground_bin_area_m2 = mirror_area_m2 / num_bins_in_mirror
    return ground_bin_area_m2


def _json_write(path, o):
    with rnw.open(path, "wt") as f:
        f.write(json_utils.dumps(o, indent=4))


def _write_versions(path, versions):
    _json_write(path=path, o=versions)


def _read_versions(path):
    with open(path, "rt") as f:
        versions = json_utils.loads(f.read())
    return versions


def _gather_versions_now():
    return {
        "magnetic_deflection": __version__,
        "corsika_primary": corsika_primary.__version__,
        "spherical_histogram": spherical_histogram.__version__,
        "spherical_coordinates": spherical_coordinates.__version__,
        "atmospheric_cherenkov_response": atmospheric_cherenkov_response.__version__,
    }


def _versions_equal(a, b):
    for key in a:
        if a[key] != b[key]:
            return False
    return True


def _population_run_job(job):
    opj = os.path.join
    sm = SkyMap(work_dir=job["work_dir"])

    corsika_steering_dict = cherenkov_pool.production.make_steering(
        run_id=job["run_id"],
        site=sm.config["site"],
        particle_id=sm.config["particle"]["corsika_particle_id"],
        particle_energy_start_GeV=sm.binning["energy"]["start"],
        particle_energy_stop_GeV=sm.binning["energy"]["stop"],
        particle_energy_power_slope=sm.config["energy_power_slope"],
        particle_cone_azimuth_rad=0.0,
        particle_cone_zenith_rad=0.0,
        particle_cone_opening_angle_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
        num_showers=job["num_showers"],
    )

    reports, cermap = cherenkov_pool.production.histogram_cherenkov_pool(
        corsika_steering_dict=corsika_steering_dict,
        binning=sm.binning,
        threshold_num_photons=sm.threshold_num_photons,
    )

    assert len(reports) == len(corsika_steering_dict["primaries"])

    stage_dir = opj(job["work_dir"], "stage")
    os.makedirs(stage_dir, exist_ok=True)

    out_path = opj(stage_dir, "{:06d}".format(job["run_id"]))

    with rnw.open(out_path + ".reports.rec", "wb") as f:
        f.write(reports.tobytes(order="c"))

    utils.write_array(
        path=out_path + ".map.exposure.rec",
        a=cermap.exposure,
    )

    utils.write_array(
        path=out_path + ".map.cherenkov_to_primary.rec",
        a=cermap.cherenkov_to_primary,
    )

    utils.write_array(
        path=out_path + ".map.primary_to_cherenkov.rec",
        a=cermap.primary_to_cherenkov,
    )

    return True


def _collect_stage_into_results(work_dir, skymap=None):
    _collect_stage_into_results_reports(work_dir=work_dir)
    _collect_stage_into_results_map(work_dir=work_dir, skymap=skymap)


def _collect_stage_into_results_map(work_dir, skymap=None):
    opj = os.path.join
    stage_dir = opj(work_dir, "stage")

    results_dir = opj(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if skymap is None:
        skymap = SkyMap(work_dir=work_dir)

    NUM_E = skymap.binning["energy"]["num"]
    NUM_S = len(skymap.binning["sky"].faces)

    names = {
        "exposure": {"dtype": "u8", "shape": (NUM_E, NUM_S)},
        "cherenkov_to_primary": {
            "dtype": "u2",
            "shape": (NUM_E, NUM_S, NUM_S),
        },
        "primary_to_cherenkov": {
            "dtype": "f4",
            "shape": (NUM_E, NUM_S, NUM_S),
        },
    }

    for name in names:
        paths = glob.glob(opj(stage_dir, "*.map.{:s}.rec".format(name)))

        out_path = opj(results_dir, "map.{:s}.rec".format(name))
        if os.path.exists(out_path):
            base = utils.read_array(path=out_path)
        else:
            base = np.zeros(
                shape=names[name]["shape"],
                dtype=names[name]["dtype"],
            )

        for path in paths:
            addon = utils.read_array(path=path)
            base = base + addon

        utils.write_array(path=out_path + ".part", a=base)
        os.rename(out_path + ".part", out_path)
        for path in paths:
            os.remove(path)


def _collect_stage_into_results_reports(work_dir):
    opj = os.path.join
    stage_dir = opj(work_dir, "stage")

    results_dir = opj(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    report_paths = glob.glob(opj(stage_dir, "*.reports.rec"))
    report_paths = sorted(report_paths)

    with tarfile.open(opj(results_dir, "reports.tar.part"), "w|") as otf:
        if os.path.exists(opj(results_dir, "reports.tar")):
            with tarfile.open(opj(results_dir, "reports.tar"), "r|") as itf:
                for tarinfo in itf:
                    payload = itf.extractfile(tarinfo).read()
                    _append_tar(otf, tarinfo.name, payload)

        for report_path in report_paths:
            with open(report_path, "rb") as f:
                payload = f.read()
            report_basename = os.path.basename(report_path)
            _append_tar(otf, report_basename + ".gz", gzip.compress(payload))
    os.rename(
        opj(results_dir, "reports.tar.part"), opj(results_dir, "reports.tar")
    )
    for report_path in report_paths:
        os.remove(report_path)


def _append_tar(tarfout, name, payload_bytes):
    tarinfo = tarfile.TarInfo()
    tarinfo.name = name
    tarinfo.size = len(payload_bytes)
    with io.BytesIO() as fileobj:
        fileobj.write(payload_bytes)
        fileobj.seek(0)
        tarfout.addfile(tarinfo=tarinfo, fileobj=fileobj)


def reports_read(path, dtype=None):
    if dtype is None:
        dtype = (
            cherenkov_pool.production.histogram_cherenkov_pool_report_dtype()
        )

    full_dtype = (
        cherenkov_pool.production.histogram_cherenkov_pool_report_dtype()
    )

    out = dynamicsizerecarray.DynamicSizeRecarray(dtype=dtype)

    with tarfile.open(path, "r|") as itf:
        for tarinfo in itf:
            payload_gz = itf.extractfile(tarinfo).read()
            payload = gzip.decompress(payload_gz)
            reports_block = np.frombuffer(buffer=payload, dtype=full_dtype)

            reports_block_out = recarray_utils.init(
                dtype=dtype,
                size=reports_block.size,
            )
            for key in dtype:
                name = key[0]
                reports_block_out[name] = reports_block[name]

            out.append_recarray(reports_block_out)

    return out


def _combine_weights_of_sky_and_ene_bins(
    skybins, skybins_weights, enebins, enebins_weights
):
    num = len(skybins) * len(enebins)
    sky = (-1) * np.ones(num, dtype=int)
    ene = (-1) * np.ones(num, dtype=int)
    weights = np.nan * np.ones(num, dtype=float)

    i = 0
    for e in range(len(enebins)):
        for s in range(len(skybins)):
            sky[i] = skybins[s]
            ene[i] = enebins[e]
            weights[i] = enebins_weights[e] * skybins_weights[s]
            i += 1

    if num > 0:
        weights = weights / np.sum(weights)

    return sky, ene, weights


def _run_job_plot_query_ball(job):
    skymap = SkyMap(work_dir=job["work_dir"])
    for call in job["calls"]:
        skymap.plot_query_ball(**call)
