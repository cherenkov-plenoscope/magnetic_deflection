"""
allsky
======
Allsky allows you to populate, store, query the statistics of atmospheric
showers and their deflection due to earth's magnetic field.
"""
import os
import json_utils
import rename_after_writing as rnw
import atmospheric_cherenkov_response
import binning_utils
import corsika_primary
from . import binning
from . import storage


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

    config = read_config(work_dir=work_dir)
    assert_config_valid(config=config)

    # storage
    # -------
    stroage.init(
        storage_dir=os.path.join(work_dir, "storage"),
        direction_num_bins=direction_num_bins,
        energy_num_bins=energy_num_bins,
    )


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
        self.work_dir = work_dir
        self.config = read_config(work_dir=work_dir)
        self.binning = binning.Binning(config=self.config["binning"])

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out
