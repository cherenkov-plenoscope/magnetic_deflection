import numpy as np
import corsika_primary as cpw
from .. import corsika
import rename_after_writing as rnw
import os
import glob
import uuid
import json_line_logger


def make_steering(
    run_id,
    site,
    particle_id,
    particle_energy_start_GeV,
    particle_energy_stop_GeV,
    particle_energy_power_slope,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
):
    assert run_id > 0
    i8 = np.int64
    f8 = np.float64

    prng = np.random.Generator(np.random.PCG64(seed=run_id))

    steering = {}
    steering["run"] = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle_energy_start_GeV * 0.99),
            "stop_GeV": f8(particle_energy_stop_GeV * 1.01),
        },
        "random_seed": cpw.random.seed.make_simple_seed(seed=run_id),
    }
    steering["primaries"] = []

    for airshower_id in np.arange(1, num_showers + 1):
        az, zd = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=np.deg2rad(particle_cone_azimuth_deg),
            zenith_rad=np.deg2rad(particle_cone_zenith_deg),
            min_scatter_opening_angle_rad=np.deg2rad(0.0),
            max_scatter_opening_angle_rad=np.deg2rad(
                particle_cone_opening_angle_deg
            ),
            max_iterations=1000,
        )
        energy_GeV = cpw.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=particle_energy_start_GeV,
            upper_limit=particle_energy_stop_GeV,
            power_slope=particle_energy_power_slope,
            num_samples=1,
        )[0]
        prm = {
            "particle_id": f8(particle_id),
            "energy_GeV": f8(energy_GeV),
            "zenith_rad": f8(zd),
            "azimuth_rad": f8(az),
            "depth_g_per_cm2": f8(0.0),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering



def make_statistics(corsika_steering_dict, cherenkov_pools):
    event_number2idx = {}
    for idx, cherenkov_pool in enumerate(cherenkov_pools):
        event_number2idx[cherenkov_pools["event"]] = idx

    outpools = []
    for event_idx, particle in corsika_steering_dict["primaries"]:
        event_number = event_idx + 1

        if event_number in event_number2idx:
            # there was enough light to carachterize the cherenkov-pool
            outpool = cherenkov_pools[event_number2idx[event_number]]
        else:
            # there was NOT enough light to carachterize the cherenkov-pool
            outpool = {
                "run": corsika_steering_dict["run_id"],
                "event": event_number,
                "particle_azimuth_deg": float(np.rad2deg(particle["azimuth_rad"])),
                "particle_zenith_deg": float(np.rad2deg(particle["zenith_rad"])),
                "particle_energy_GeV": float(particle["energy_GeV"]),
                "cherenkov_num_photons": float("nan"),
                "cherenkov_num_bunches": float("nan"),
                "cherenkov_x_m": float("nan"),
                "cherenkov_y_m": float("nan"),
                "cherenkov_radius50_m": float("nan"),
                "cherenkov_cx_rad": float("nan"),
                "cherenkov_cy_rad": float("nan"),
                "cherenkov_angle50_rad": float("nan"),
                "cherenkov_t_s": float("nan"),
                "cherenkov_t_std_s": float("nan"),
            }
        outpools.append(outpool)
    return outpools


def init(production_dir):
    os.makedirs(production_dir, exist_ok=True)
    with rnw.open(os.path.join(production_dir, "lock.open"), "wt") as f:
        pass

    with rnw.open(os.path.join(production_dir, "next_run_id.txt"), "wt") as f:
        f.write("1")

    os.makedirs(os.path.join(production_dir, "logs"), exist_ok=True)


class Production:
    def __init__(self, production_dir):
        self.production_dir = production_dir
        self.uuid = str(uuid.uuid4())
        self.log = json_line_logger.LoggerFile(
            path=os.path.join(
                self.production_dir, "logs", "{:s}.jsonl".format(self.uuid)
            ),
            name=self.uuid,
        )

    def _read_lock_uuid(self):
        lock_paths = glob.glob(os.path.join(self.production_dir, "lock.*"))
        assert len(lock_paths) == 1
        lock_basename = os.path.basename(lock_paths[0])
        lock_uuid = str.split(lock_basename, ".")[1]
        return lock_uuid

    def lock_is_mine(self):
        lock_uuid = self._read_lock_uuid()
        if lock_uuid == self.uuid:
            return True
        else:
            return False

    def lock_is_open(self):
        lock_uuid = self._read_lock_uuid()
        if lock_uuid == "open":
            return True
        else:
            return False

    def lock(self):
        self.log.info("Try to get lock.")
        if not self.lock_is_open():
            msg = "Can not get lock becasue lock is not open."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can get lock because it is open.")

        os.rename(
            os.path.join(self.production_dir, "lock.open"),
            os.path.join(self.production_dir, "lock.{:s}".format(self.uuid)),
        )
        self.log.info("Lock is mine.")

    def unlock(self):
        self.log.info("Try to open lock.")
        if not self.lock_is_mine():
            msg = "Can not open lock becasue lock is not mine."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can open lock because it is mine.")

        os.rename(
            os.path.join(self.production_dir, "lock.{:s}".format(self.uuid)),
            os.path.join(self.production_dir, "lock.open"),
        )
        self.log.info("Lock is open.")

    def get_next_run_id_and_bumb(self):
        self.log.info("Try to bumb run_id.")
        if not self.lock_is_mine():
            msg = "The lock is not mine. I can not bumb the run_id."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can bumb run_id because the lock is mine")

        path = os.path.join(self.production_dir, "next_run_id.txt")
        with open(path, "rt") as f:
            next_run_id = int(f.read())

        self.log.info("Next run_id is {:d}.".format(next_run_id))

        next_next_run_id = next_run_id + 1
        with open(path, "wt") as f:
            f.write("{:d}".format(next_next_run_id))

        self.log.info(
            "Bumbed the next next run_id to {:d}.".format(next_next_run_id)
        )

        return next_run_id
