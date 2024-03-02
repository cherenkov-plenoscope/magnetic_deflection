from .. import utils
from ..version import __version__
from .. import allsky
from .. import cherenkov_pool

import os
import numpy as np
import copy

import atmospheric_cherenkov_response
import rename_after_writing as rnw
import json_utils
import solid_angle_utils
import spherical_histogram
import spherical_coordinates
import binning_utils
import corsika_primary
import triangle_mesh_io


def init(
    work_dir,
    particle_key,
    site_key,
    energy_bin_edges_GeV,
    altitude_bin_edges_m,
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

    # altitude
    # --------
    assert binning_utils.is_strictly_monotonic_increasing(altitude_bin_edges_m)
    _json_write(
        path=opj(binning_dir, "altitude_bin_edges_m.json"),
        o=altitude_bin_edges_m,
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


def init_example(work_dir):
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

    return init(
        work_dir=work_dir,
        particle_key="electron",
        site_key="lapalma",
        energy_bin_edges_GeV=_guess_energy_bin_edges_GeV(),
        altitude_bin_edges_m=_guess_cherenkov_altitude_p50_bin_edges_m(),
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
        self.binning["altitude"] = binning_utils.Binning(
            bin_edges=cfg["binning"]["altitude_bin_edges_m"]
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
            results = pool.map(_population_run_job, jobs)
            self.store.commit_stage(pool=pool)

        production_lock.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out


def _guess_cherenkov_altitude_p50_bin_edges_m():
    return np.geomspace(2**10, 2**16, 3 + 1)


def _guess_energy_bin_edges_GeV():
    return np.geomspace(2 ** (-2), 2 ** (6), 32 + 1)


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
        particle_energy_start_GeV=sm.energy["start"],
        particle_energy_stop_GeV=sm.energy["stop"],
        particle_energy_power_slope=-2.0,
        particle_cone_azimuth_rad=0.0,
        particle_cone_zenith_rad=0.0,
        particle_cone_opening_angle_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
        num_showers=job["num_showers"],
    )

    pools, cer_to_par_map = cherenkov_pool.production.histogram_cherenkov_pool(
        corsika_steering_dict=corsika_steering_dict,
        binning=sm.binning,
        threshold_num_photons=sm.threshold_num_photons,
    )

    assert len(pools) == len(corsika_steering_dict["primaries"])
    stage_dir = opj(job["work_dir"], "stage")

    pools_path = opj(stage_dir, "{:06d}_pools.jsonl".format(job["run_id"]))
    json_utils.lines.write(path=pools_path, obj_list=pools)

    overflow_path = opj(
        stage_dir, "{:06d}_overflow.jsonl".format(job["run_id"])
    )
    json_utils.lines.write(
        path=overflow_path, obj_list=cer_to_par_map.overflow
    )

    exposure_path = opj(stage_dir, "{:06d}_exposure.u8".format(job["run_id"]))
    with rnw.open(exposure_path, "wb") as f:
        f.write(cer_to_par_map.exposure.tobytes(order="c"))

    return True
