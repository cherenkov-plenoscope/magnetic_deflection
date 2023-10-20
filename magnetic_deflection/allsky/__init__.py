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

    def populate(self, num_showers=1000):
        assert num_showers > 0
        assert os.path.exists(self.config["corsika_primary"]["path"])

        self.production.lock()

        corsika_steering_dict = production.make_steering(
            run_id=self.production.get_next_run_id_and_bumb(),
            site=self.config["site"],
            particle_id=self.config["particle"]["corsika_particle_id"],
            particle_energy_start_GeV=self.binning.energy["start"],
            particle_energy_stop_GeV=self.binning.energy["stop"],
            particle_energy_power_slope=-2.0,
            particle_cone_azimuth_deg=0.0,
            particle_cone_zenith_deg=0.0,
            particle_cone_opening_angle_deg=self.config["binning"][
                "direction"
            ]["particle_max_zenith_distance_deg"],
            num_showers=num_showers,
        )

        showers = production.estimate_cherenkov_pool(
            corsika_primary_path=self.config["corsika_primary"]["path"],
            corsika_steering_dict=corsika_steering_dict,
            min_num_cherenkov_photons=self.config["cherenkov_pool_statistics"][
                "min_num_cherenkov_photons"
            ],
        )
        assert len(showers) == len(corsika_steering_dict["primaries"])

        # staging
        # -------
        cherenkov_stage = self.store.make_empty_stage()
        particle_stage = self.store.make_empty_stage()

        num_not_enough_light = 0

        for shower in showers:
            if (
                shower["cherenkov_num_photons"]
                >= self.config["cherenkov_pool_statistics"][
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
                ) = self.binning.query(
                    azimuth_deg=cer_az_deg,
                    zenith_deg=cer_zd_deg,
                    energy_GeV=shower["particle_energy_GeV"],
                )
                # print("cer", shower["run"], shower["event"], dbin, ebin)
                valid_bin = self.binning.is_valid_dbin_ebin(
                    dbin=dbin, ebin=ebin
                )
                if delta_phi_deg > 10.0 or not valid_bin:
                    msg = ""
                    msg += "Warning: Shower ({:d},{:d}) is ".format(
                        shower["run"], shower["event"]
                    )
                    msg += (
                        "{:f}deg off the closest cherenkov-bin-center".format(
                            delta_phi_deg
                        )
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
            (delta_phi_deg, delta_energy), (dbin, ebin) = self.binning.query(
                azimuth_deg=shower["particle_azimuth_deg"],
                zenith_deg=shower["particle_zenith_deg"],
                energy_GeV=shower["particle_energy_GeV"],
            )
            # print("par", shower["run"], shower["event"], dbin, ebin)
            valid_bin = self.binning.is_valid_dbin_ebin(dbin=dbin, ebin=ebin)
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
                particle_stage["showers"][dbin][ebin].append(
                    copy.deepcopy(shower)
                )

        print(
            "not_enough_light: {:d} of {:d}".format(
                num_not_enough_light, len(showers)
            )
        )

        # write to stage
        # --------------
        self.store.add_cherenkov_to_stage(cherenkov_stage=cherenkov_stage)
        self.store.add_particle_to_stage(particle_stage=particle_stage)

        self.production.unlock()

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out
