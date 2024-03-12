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

    # run_id
    # ------
    allsky.production.init(production_dir=opj(work_dir, "production"))

    return SkyMap(work_dir=work_dir)


def init_example(work_dir, particle_key="electron", site_key="lapalma"):
    PORTAL_FOV_HALF_ANGLE_RAD = np.deg2rad(3.25)
    PORTAL_MIRROR_DIAMETER_M = 71.0
    overhead = 2.0

    vertices, faces = _guess_sky_vertices_and_faces(
        fov_half_angle_rad=PORTAL_FOV_HALF_ANGLE_RAD,
        num_faces_in_fov=overhead,
        max_zenith_distance_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
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

    def _query_ball(self, query, map_key):
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
        num_sky_bins = len(self.binning["sky"].faces)

        mmm = self.map_primary_to_cherenkov_normalized_per_sr()

        sky_intensity = np.zeros(num_sky_bins)

        for i in range(len(fsky)):
            skybin = fsky[i]
            enebin = fene[i]
            weight_bin = fweights[i]

            sky_intensity_bin = np.zeros(num_sky_bins)
            for jbin in range(num_sky_bins):
                if map_key == "primary_to_cherenkov":
                    sky_intensity_bin[jbin] = mmm[enebin][skybin][jbin]
                elif map_key == "cherenkov_to_primary":
                    sky_intensity_bin[jbin] = mmm[enebin][jbin][skybin]
                else:
                    raise ValueError("Unknown map_key '{:s}'.".format(map_key))

            sky_intensity += weight_bin * sky_intensity_bin

        return sky_intensity

    def query_ball_primary_to_cherenkov(self, query):
        return self._query_ball(query=query, map_key="primary_to_cherenkov")

    def query_ball_cherenkov_to_primary(self, query):
        return self._query_ball(query=query, map_key="cherenkov_to_primary")

    def plot_query_ball_primary_to_cherenkov(
        self,
        query,
        path,
    ):
        sky_values = self.query_ball_primary_to_cherenkov(query=query)

        plotting.plot_query(
            path=path,
            num_pixel=1280,
            query=query,
            skymap=self,
            sky_values=sky_values,
            sky_values_min=1e3,
            sky_values_max=1e8,
            sky_values_label="cherenkov density / (sr)"
            + svgplt.text.superscript("-1"),
            sky_values_scale="log",
            sky_mask=None,
            sky_mask_color="blue",
            colormap_name="inferno",
        )

        return sky_values

    def plot_query_ball_cherenkov_to_primary(
        self,
        query,
        path,
        threshold_cherenkov_density_per_sr=1e4,
    ):
        sky_values = self.query_ball_cherenkov_to_primary(query=query)
        sky_mask = sky_values >= threshold_cherenkov_density_per_sr

        sky_mask = (
            spherical_histogram.mesh.fill_faces_mask_if_two_neighbors_true(
                faces_mask=sky_mask,
                faces_neighbors=self.binning["sky"].faces_neighbors,
            )
        )

        plotting.plot_query(
            path=path,
            num_pixel=1280,
            query=query,
            skymap=self,
            sky_values=sky_values,
            sky_values_min=1e3,
            sky_values_max=1e8,
            sky_values_label="cherenkov density / (sr)"
            + svgplt.text.superscript("-1"),
            sky_values_scale="log",
            sky_mask=sky_mask,
            sky_mask_color="red",
            colormap_name="viridis",
        )

        return sky_values

    def demonstrate(
        self,
        path,
        pool,
        num_jobs=6,
        queries=None,
        map_key="primary_to_cherenkov",
        threshold_cherenkov_density_per_sr=1e4,
    ):
        os.makedirs(path, exist_ok=True)
        if queries is None:
            queries = querying.example(
                min_energy_GeV=self.binning["energy"]["limits"][0],
                max_energy_GeV=self.binning["energy"]["limits"][1],
                num=6,
            )

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
                    imgpath = os.path.join(path, name + "." + map_key + ".jpg")
                    if not os.path.exists(imgpath):
                        call = {}
                        call["query"] = queries[i]
                        call["map_key"] = map_key
                        call["path"] = imgpath
                        call[
                            "threshold_cherenkov_density_per_sr"
                        ] = threshold_cherenkov_density_per_sr
                        job["calls"].append(call)
            jobs.append(job)

        print(len(jobs))
        pool.map(_run_job_plot_query_ball, jobs)

        try:
            from sebastians_matplotlib_addons import video

            video.write_video_from_image_slices(
                image_sequence_wildcard_path=os.path.join(
                    path, "%06d" + "." + map_key + ".jpg"
                ),
                output_path=os.path.join(path, map_key + ".mov"),
            )
        except:
            pass

    def map_primary_to_cherenkov(self):
        return utils.read_array(
            path=os.path.join(
                self.work_dir, "results", "map.primary_to_cherenkov.rec"
            )
        )

    def map_exposure(self):
        if not hasattr(self, "_map_exposure"):
            self._map_exposure = utils.read_array(
                path=os.path.join(self.work_dir, "results", "map.exposure.rec")
            )
        return self._map_exposure

    def map_primary_to_cherenkov_normalized_per_sr(self):
        if not hasattr(self, "_map_primary_to_cherenkov_normalized_per_sr"):
            prm2cer = self.map_primary_to_cherenkov()
            num_prm = self.map_exposure()

            self._map_primary_to_cherenkov_normalized_per_sr = np.zeros(
                shape=prm2cer.shape,
                dtype=np.float32,
            )

            for enebin in range(prm2cer.shape[0]):
                for prmbin in range(prm2cer.shape[1]):
                    eee = num_prm[enebin][prmbin]
                    if eee > 0:
                        vvv = prm2cer[enebin][prmbin].copy()
                        vvv /= eee
                        vvv /= self.binning["sky"].faces_solid_angles
                        self._map_primary_to_cherenkov_normalized_per_sr[
                            enebin
                        ][prmbin] = vvv

        # cross check
        # -----------
        for enebin in range(
            self._map_primary_to_cherenkov_normalized_per_sr.shape[0]
        ):
            for prmbin in range(
                self._map_primary_to_cherenkov_normalized_per_sr.shape[1]
            ):
                if num_prm[enebin][prmbin] == 0:
                    assert (
                        np.sum(
                            self._map_primary_to_cherenkov_normalized_per_sr[
                                enebin
                            ][prmbin]
                        )
                        == 0.0
                    )

        return self._map_primary_to_cherenkov_normalized_per_sr


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

    reports, cerskymap = cherenkov_pool.production.histogram_cherenkov_pool(
        corsika_steering_dict=corsika_steering_dict,
        binning=sm.binning,
    )

    assert len(reports) == len(corsika_steering_dict["primaries"])

    stage_dir = opj(job["work_dir"], "stage")
    os.makedirs(stage_dir, exist_ok=True)

    out_path = opj(stage_dir, "{:06d}".format(job["run_id"]))

    with rnw.open(out_path + ".reports.rec", "wb") as f:
        f.write(reports.tobytes(order="c"))

    utils.write_array(
        path=out_path + ".map.exposure.rec",
        a=cerskymap.exposure,
    )

    utils.write_array(
        path=out_path + ".map.primary_to_cherenkov.rec",
        a=cerskymap.primary_to_cherenkov,
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


def reports_read(path, dtype=None, query=None):
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

            if query is not None:
                mask = reports_query_to_mask(
                    reports=reports_block,
                    query=query,
                )
                reports_block = reports_block[mask]

            reports_block_out = recarray_utils.init(
                dtype=dtype,
                size=reports_block.size,
            )
            for key in dtype:
                name = key[0]
                reports_block_out[name] = reports_block[name]

            out.append_recarray(reports_block_out)

    return out.to_recarray()


def reports_query_to_mask(reports, query):
    query_cx, query_cy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=query["azimuth_rad"],
        zenith_rad=query["zenith_rad"],
    )
    delta_rad = spherical_coordinates.angle_between_cx_cy(
        cx1=query_cx,
        cy1=query_cy,
        cx2=reports["particle_cx"],
        cy2=reports["particle_cy"],
    )
    mask_cone = delta_rad <= query["half_angle_rad"]
    mask_energy = np.logical_and(
        reports["particle_energy_GeV"] >= query["energy_start_GeV"],
        reports["particle_energy_GeV"] < query["energy_stop_GeV"],
    )
    return np.logical_and(mask_cone, mask_energy)


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
        if call["map_key"] == "primary_to_cherenkov":
            skymap.plot_query_ball_primary_to_cherenkov(
                query=call["query"],
                path=call["path"],
            )
        elif call["map_key"] == "cherenkov_to_primary":
            skymap.plot_query_ball_cherenkov_to_primary(
                query=call["query"],
                path=call["path"],
                threshold_cherenkov_density_per_sr=call[
                    "threshold_cherenkov_density_per_sr"
                ],
            )
    return True
