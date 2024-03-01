from .. import utils
from ..version import __version__
from .. import allsky

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
    threshold_photons_per_sr,
    sky_vertices,
    sky_faces,
):
    opj = os.path.join

    os.makedirs(work_dir, exist_ok=True)
    _write_versions(
        path=opj(work_dir, "versions.json"),
        versions=_gather_versions_now(),
    )

    config_dir = os.path.join(work_dir, "config")
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
        config = json_utils.tree.read(path=opj(work_dir, "config"))
        conbin = config["binning"]

        self.energy = binning_utils.Binning(
            bin_edges=conbin["energy_bin_edges_GeV"]
        )
        self.altitude = binning_utils.Binning(
            bin_edges=conbin["altitude_bin_edges_m"]
        )
        self.threshold_photons_per_sr = conbin["threshold_photons_per_sr"][
            "threshold_photons_per_sr"
        ]

        with open(opj(work_dir, "config", "binning", "sky.obj"), "rt") as f:
            _sky_obj = triangle_mesh_io.obj.loads(f.read())

        (
            _sky_vertices,
            _sky_faces,
        ) = spherical_histogram.mesh.obj_to_vertices_and_faces(obj=_sky_obj)

        self.sky = spherical_histogram.geometry.HemisphereGeometry(
            vertices=_sky_vertices,
            faces=_sky_faces,
        )

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out


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
    num_vertices=511, max_zenith_distance_rad=np.deg2rad(89)
):
    vertices = spherical_histogram.mesh.make_vertices(
        num_vertices=num_vertices,
        max_zenith_distance_rad=max_zenith_distance_rad,
    )
    faces = spherical_histogram.mesh.make_faces(vertices=vertices)
    return vertices, faces


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
