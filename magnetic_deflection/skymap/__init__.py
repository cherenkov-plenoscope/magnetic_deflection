from .. import utils
from ..version import __version__
from .. import allsky
from .. import cherenkov_pool
from . import recarray_utils

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
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_start_GeV,
        energy_stop_GeV,
    ):
        faces, faces_weights = self.binning[
            "sky"
        ].query_cone_weiths_azimuth_zenith(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
        )

        enebins = binning_utils.query_ball(
            bin_edges=self.binning["energy"]["edges"],
            start=energy_start_GeV,
            stop=energy_stop_GeV,
        )
        return faces, enebins

    def query_ball_primary_to_cherenkov(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_start_GeV,
        energy_stop_GeV,
        path=None,
        vmax=1e6,
    ):
        faces, enebins = self._query_ball_bins(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
        )

        print(faces, enebins)
        ex = self.map_exposure()
        p2c = self.map_primary_to_cherenkov()

        num_bins = 0
        sky_intensity = np.zeros(len(self.binning["sky"].faces))
        for enebin in enebins:
            for face in faces:
                num_bins += 1
                sky_intensity += p2c[enebin][face] / ex[enebin][face]

        sky_intensity /= num_bins

        if path is not None:
            # plot
            # ----

            vmin = 0
            vmax = np.max(sky_intensity)

            fig = svgplt.Fig(cols=1080, rows=1080)
            ax = {}
            ax = svgplt.hemisphere.Ax(fig=fig)
            ax["span"] = (0.1, 0.1, 0.8, 0.8)

            fov_ring_verts_uxyz = allsky.viewcone.make_ring(
                half_angle_rad=half_angle_rad,
                endpoint=True,
                fn=137,
            )
            fov_ring_verts_uxyz = allsky.viewcone.rotate(
                vertices_uxyz=fov_ring_verts_uxyz,
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                mount="cable_robot",
            )

            cmap = svgplt.color.Map(
                "inferno",
                start=vmin,
                stop=vmax,
            )

            mesh_look = svgplt.hemisphere.init_mesh_look(
                num_faces=len(self.binning["sky"].faces),
                stroke=None,
                fill=svgplt.color.css("black"),
                fill_opacity=1.0,
            )

            for i in range(len(self.binning["sky"].faces)):
                mesh_look["faces_fill"][i] = cmap(sky_intensity[i])

            svgplt.hemisphere.ax_add_mesh(
                ax=ax,
                vertices=self.binning["sky"].vertices,
                faces=self.binning["sky"].faces,
                max_radius=1.0,
                **mesh_look,
            )

            svgplt.ax_add_path(
                ax=ax,
                xy=fov_ring_verts_uxyz[:, 0:2],
                stroke=svgplt.color.css("red"),
                fill=None,
            )
            svgplt.hemisphere.ax_add_grid(ax=ax)

            svgplt.ax_add_text(
                ax=ax,
                xy=[-0.6, 1.15],
                text="primary to cherenkov map",
                fill=svgplt.color.css("black"),
                font_family="math",
                font_size=30,
            )
            svgplt.ax_add_text(
                ax=ax,
                xy=[0.3, 1.0],
                text="energy: {: 8.3f} to {: 8.3}GeV".format(
                    energy_start_GeV,
                    energy_stop_GeV,
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

    def query_ball_cherenkov_to_primary(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_start_GeV,
        energy_stop_GeV,
        quantile,
        path=None,
    ):
        faces, enebins = self._query_ball_bins(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
        )

        print(faces, enebins)
        ex = self.map_exposure()
        c2p = self.map_cherenkov_to_primary()

        num_bins = 0
        sky_intensity = np.zeros(len(self.binning["sky"].faces))
        for enebin in enebins:
            for face in faces:
                num_bins += 1
                sky_intensity += c2p[enebin][face] / ex[enebin][face]

        sky_intensity /= num_bins

        assert 0.0 <= quantile <= 1.0

        face_order = np.argsort((-1) * sky_intensity)
        part = 0.0
        target = quantile * np.sum(sky_intensity)
        face_quantile_mask = np.zeros(sky_intensity.shape[0], dtype=bool)
        for ii in range(len(face_order)):
            print(sky_intensity[face_order[ii]])
            if part + sky_intensity[face_order[ii]] < target:
                part += sky_intensity[face_order[ii]]
                face_quantile_mask[face_order[ii]] = True
            else:
                break

        if path is not None:
            # plot
            # ----

            vmin = 0
            vmax = np.max(sky_intensity)

            fig = svgplt.Fig(cols=1080, rows=1080)
            ax = {}
            ax = svgplt.hemisphere.Ax(fig=fig)
            ax["span"] = (0.1, 0.1, 0.8, 0.8)

            fov_ring_verts_uxyz = allsky.viewcone.make_ring(
                half_angle_rad=half_angle_rad,
                endpoint=True,
                fn=137,
            )
            fov_ring_verts_uxyz = allsky.viewcone.rotate(
                vertices_uxyz=fov_ring_verts_uxyz,
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                mount="cable_robot",
            )

            cmap = svgplt.color.Map(
                "viridis",
                start=vmin,
                stop=vmax,
            )

            mesh_look = svgplt.hemisphere.init_mesh_look(
                num_faces=len(self.binning["sky"].faces),
                stroke=None,
                fill=svgplt.color.css("black"),
                fill_opacity=1.0,
            )

            for i in range(len(self.binning["sky"].faces)):
                mesh_look["faces_fill"][i] = cmap(sky_intensity[i])
                if face_quantile_mask[i]:
                    mesh_look["faces_stroke"][i] = svgplt.color.css("red")

            svgplt.hemisphere.ax_add_mesh(
                ax=ax,
                vertices=self.binning["sky"].vertices,
                faces=self.binning["sky"].faces,
                max_radius=1.0,
                **mesh_look,
            )

            svgplt.ax_add_path(
                ax=ax,
                xy=fov_ring_verts_uxyz[:, 0:2],
                stroke=svgplt.color.css("red"),
                fill=None,
            )
            svgplt.hemisphere.ax_add_grid(ax=ax)

            svgplt.ax_add_text(
                ax=ax,
                xy=[-0.6, 1.15],
                text="primary to cherenkov map",
                fill=svgplt.color.css("black"),
                font_family="math",
                font_size=30,
            )
            svgplt.ax_add_text(
                ax=ax,
                xy=[0.3, 1.0],
                text="energy: {: 8.3f} to {: 8.3}GeV".format(
                    energy_start_GeV,
                    energy_stop_GeV,
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
    return binning_utils.powerspace(
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
