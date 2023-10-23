import os
import uuid
import glob
import numpy as np
import rename_after_writing as rnw
from . import dynamicsizerecarray


DIRECTION_BIN_DIR_STR = "{direction_bin:06d}"
ENERGY_BIN_DIR_STR = "{energy_bin:06d}"


def init(store_dir, energy_num_bins, direction_num_bins):
    assert direction_num_bins >= 1
    assert energy_num_bins >= 1

    os.makedirs(store_dir, exist_ok=True)

    for d_bin in range(direction_num_bins):
        d_dir = os.path.join(
            store_dir, DIRECTION_BIN_DIR_STR.format(direction_bin=d_bin)
        )
        os.makedirs(d_dir, exist_ok=True)

        for e_bin in range(energy_num_bins):
            d_e_dir = os.path.join(
                d_dir, ENERGY_BIN_DIR_STR.format(energy_bin=e_bin)
            )
            os.makedirs(d_e_dir, exist_ok=True)

            showers_write(
                path=os.path.join(d_e_dir, "cherenkov.rec"),
                showers=showers_init(size=0),
            )
            os.makedirs(
                os.path.join(d_e_dir, "cherenkov_stage"), exist_ok=True
            )

            showers_write(
                path=os.path.join(d_e_dir, "particle.rec"),
                showers=showers_init(size=0),
            )
            os.makedirs(os.path.join(d_e_dir, "particle_stage"), exist_ok=True)


class Store:
    def __init__(self, store_dir, energy_num_bins, direction_num_bins):
        assert energy_num_bins >= 1
        assert direction_num_bins >= 1

        self.store_dir = store_dir
        self.energy_num_bins = energy_num_bins
        self.direction_num_bins = direction_num_bins

    def __repr__(self):
        out = "{:s}(num bins: energy={:d}, direction={:d})".format(
            self.__class__.__name__,
            self.energy_num_bins,
            self.direction_num_bins,
        )
        return out

    def bin_dir(self, direction_bin, energy_bin):
        """
        Returns the path of a direction-energy-bin.
        """
        self.assert_bin_indices(
            direction_bin=direction_bin, energy_bin=energy_bin
        )
        return os.path.join(
            self.store_dir,
            DIRECTION_BIN_DIR_STR.format(direction_bin=direction_bin),
            ENERGY_BIN_DIR_STR.format(energy_bin=energy_bin),
        )

    def assert_bin_indices(self, direction_bin, energy_bin):
        assert 0 <= direction_bin < self.direction_num_bins
        assert 0 <= energy_bin < self.energy_num_bins

    def _num_showers_in_file(self, direction_bin, energy_bin, filename):
        return showers_num_in_file(
            path=os.path.join(
                self.bin_dir(
                    direction_bin=direction_bin,
                    energy_bin=energy_bin,
                ),
                filename,
            )
        )

    def _population(self, key):
        pop = np.zeros(
            shape=(self.direction_num_bins, self.energy_num_bins),
            dtype=np.uint64,
        )
        for dbin in range(self.direction_num_bins):
            for ebin in range(self.energy_num_bins):
                pop[dbin, ebin] = self._num_showers_in_file(
                    dbin,
                    ebin,
                    filename="{key:s}.rec".format(key=key),
                )
        return pop

    def population_cherenkov(self):
        return self._population(key="cherenkov")

    def population_particle(self):
        return self._population(key="particle")

    def make_empty_stage(self):
        stage = {}
        stage["uuid"] = str(uuid.uuid4())
        stage["showers"] = []
        for dbin in range(self.direction_num_bins):
            stage["showers"].append([])

        for dbin in range(self.direction_num_bins):
            for ebin in range(self.energy_num_bins):
                stage["showers"][dbin].append([])
        return stage

    def add_cherenkov_to_stage(self, cherenkov_stage):
        self._add_to_stage(stage=cherenkov_stage, key="cherenkov")

    def add_particle_to_stage(self, particle_stage):
        self._add_to_stage(stage=particle_stage, key="particle")

    def _add_to_stage(self, stage, key):
        for dbin in range(self.direction_num_bins):
            for ebin in range(self.energy_num_bins):
                showers_records = stage["showers"][dbin][ebin]
                if len(showers_records) > 0:
                    showers_dyn = dynamicsizerecarray.DynamicSizeRecarray(
                        dtype=showers_dtype()
                    )
                    showers_dyn.append_records(showers_records)
                    showers = showers_dyn.to_recarray()

                    stage_path = os.path.join(
                        self.bin_dir(direction_bin=dbin, energy_bin=ebin),
                        "{:s}_stage".format(key),
                        "{uuid:s}.rec".format(uuid=stage["uuid"]),
                    )
                    showers_write(path=stage_path, showers=showers)

    def commit_stage(self):
        num = {}
        for key in ["cherenkov", "particle"]:
            num[key] = 0
            for dbin in range(self.direction_num_bins):
                for ebin in range(self.energy_num_bins):
                    num[key] += self._commit_stage_dbin_ebin(
                        dbin=dbin, ebin=ebin, key=key
                    )
        return num

    def _commit_stage_dbin_ebin(self, dbin, ebin, key):
        bin_dir = self.bin_dir(direction_bin=dbin, energy_bin=ebin)
        stage_dir = os.path.join(bin_dir, "{:s}_stage".format(key))
        showers_path = os.path.join(bin_dir, "{:s}.rec".format(key))
        stage_paths = glob.glob(os.path.join(stage_dir, "*.rec"))

        showers_dyn = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=showers_dtype()
        )

        for stage_path in stage_paths:
            additional_showers = showers_read(path=stage_path)
            showers_dyn.append_recarray(additional_showers)
            # os.rename(stage_path, stage_path + ".consumed")
            os.remove(stage_path)
        num_showers_in_stage = int(showers_dyn.size)

        if os.path.exists(showers_path):
            existing_showers = showers_read(path=showers_path)
            showers_dyn.append_recarray(existing_showers)

        showers = showers_dyn.to_recarray()
        showers_write(path=showers_path, showers=showers)
        return num_showers_in_stage


def showers_write(path, showers):
    assert showers.dtype == showers_dtype()
    with rnw.open(path, "wb") as f:
        f.write(showers.tobytes())


def showers_read(path):
    with open(path, "rb") as f:
        showers = np.fromstring(f.read(), dtype=showers_dtype())
    return showers


def showers_num_in_file(path):
    stat = os.stat(path)
    size_in_bytes = stat.st_size
    return size_in_bytes // shower_record_size_in_bytes()


def showers_dtype():
    return [
        ("run", "u4"),
        ("event", "u4"),
        ("particle_azimuth_deg", "f4"),
        ("particle_zenith_deg", "f4"),
        ("particle_energy_GeV", "f4"),
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_num_bunches", "f4"),
        ("cherenkov_x_m", "f4"),
        ("cherenkov_y_m", "f4"),
        ("cherenkov_radius50_m", "f4"),
        ("cherenkov_cx_rad", "f4"),
        ("cherenkov_cy_rad", "f4"),
        ("cherenkov_angle50_rad", "f4"),
        ("cherenkov_t_s", "f4"),
        ("cherenkov_t_std_s", "f4"),
    ]


def showers_init(size=0):
    return np.core.records.recarray(
        shape=size,
        dtype=showers_dtype(),
    )


def shower_record_size_in_bytes():
    rr = np.core.records.recarray(
        shape=1,
        dtype=showers_dtype(),
    )
    return len(rr.tobytes())
