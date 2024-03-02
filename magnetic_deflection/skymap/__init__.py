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
    threshold_num_photons_per_sr,
    sky_vertices,
    sky_faces,
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

    # threshold
    # ---------
    assert threshold_photons_per_sr >= 0.0
    _json_write(
        path=opj(binning_dir, "threshold_photons_per_sr.json"),
        o={"threshold_photons_per_sr": threshold_photons_per_sr},
    )

    # run_id
    # ------
    allsky.production.init(production_dir=opj(work_dir, "production"))

    return SkyMap(work_dir=work_dir)


def init_example(work_dir):
    vertices, faces = _default_sky_bin_geometry_vertices_and_faces()
    return init(
        work_dir=work_dir,
        particle_key="electron",
        site_key="lapalma",
        energy_bin_edges_GeV=_guess_energy_bin_edges_GeV(),
        altitude_bin_edges_m=_guess_cherenkov_altitude_p50_bin_edges_m(),
        threshold_photons_per_sr=_guess_threshold_photons_per_sr_for_portal_cherenkov_plenoscope(),
        sky_vertices=vertices,
        sky_faces=faces,
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
        self.threshold_photons_per_sr = conbin["threshold_photons_per_sr"][
            "threshold_photons_per_sr"
        ]

        with open(opj(work_dir, "config", "binning", "sky.obj"), "rt") as f:
            _sky_obj = triangle_mesh_io.obj.loads(f.read())

        (
            _sky_vertices,
            _sky_faces,
        ) = spherical_histogram.mesh.obj_to_vertices_and_faces(obj=_sky_obj)

        self.binning = Binning(
            energy_bin_edges_GeV=self.config["binning"][
                "energy_bin_edges_GeV"
            ],
            altitude_bin_edges_m=self.config["binning"][
                "altitude_bin_edges_m"
            ],
            sky_vertices=_sky_vertices,
            sky_faces=_sky_faces,
        )

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out


class Binning:
    def __init__(
        self,
        energy_bin_edges_GeV,
        altitude_bin_edges_m,
        sky_vertices,
        sky_faces,
    ):
        self.energy = binning_utils.Binning(bin_edges=energy_bin_edges_GeV)
        self.altitude = binning_utils.Binning(bin_edges=altitude_bin_edges_m)
        self.sky = spherical_histogram.geometry.HemisphereGeometry(
            vertices=sky_vertices,
            faces=sky_faces,
        )


def _guess_cherenkov_altitude_p50_bin_edges_m():
    return np.geomspace(2**10, 2**16, 3 + 1)


def _guess_energy_bin_edges_GeV():
    return np.geomspace(2 ** (-2), 2 ** (6), 32 + 1)


def _guess_threshold_photons_per_sr_for_portal_cherenkov_plenoscope():
    num_photons_in_fov = 25
    HALF_ANGLE_PORTAL_DEG = 3.25
    fov_solid_angle_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=np.deg2rad(HALF_ANGLE_PORTAL_DEG)
    )
    return num_photons_in_fov / fov_solid_angle_sr


def _default_sky_bin_geometry_vertices_and_faces(
    max_zenith_distance_rad=np.deg2rad(89),
):
    HALF_ANGLE_PORTAL_DEG = 3.25
    sky_solid_angle_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=max_zenith_distance_rad
    )
    fov_solid_angle_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=np.deg2rad(HALF_ANGLE_PORTAL_DEG)
    )
    NUM_FACES_IN_FOV = 2
    num_faces = NUM_FACES_IN_FOV * sky_solid_angle_sr / fov_solid_angle_sr

    APPROX_NUM_VERTICES_PER_FACE = 0.5
    num_vertices = APPROX_NUM_VERTICES_PER_FACE * num_faces

    num_vertices = int(np.ceil(num_vertices))

    vertices = spherical_histogram.mesh.make_vertices(
        num_vertices=num_vertices,
        max_zenith_distance_rad=max_zenith_distance_rad,
    )
    faces = spherical_histogram.mesh.make_faces(vertices=vertices)
    return vertices, faces


def _default_ground_bin_area():
    MIRROR_RADIUS_PORTAL = 71 / 2
    mirror_area_m2 = np.pi * MIRROR_RADIUS_PORTAL**2
    NUM_BINS_IN_MIRROR = 2
    ground_bin_area_m2 = mirror_area_m2 / NUM_BINS_IN_MIRROR
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
        num_showers=job["num_showers_per_job"],
    )

    pools, cer_to_par_map = cherenkov_pool.production.histogram_cherenkov_pool(
        corsika_steering_dict=corsika_steering_dict,
        binning=sm.binning,
        threshold_photons_per_sr=sm.threshold_photons_per_sr,
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
