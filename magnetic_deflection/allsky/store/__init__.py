import os
import uuid
import glob
import numpy as np
import json_numpy
import rename_after_writing as rnw
from .. import dynamicsizerecarray
from . import page


DIRECTION_BIN_DIR_STR = "{dir_bin:06d}"
ENERGY_BIN_DIR_STR = "{ene_bin:06d}"


def init(store_dir, num_ene_bins, num_dir_bins):
    num_dir_bins = int(num_dir_bins)
    num_ene_bins = int(num_ene_bins)
    assert num_dir_bins >= 1
    assert num_ene_bins >= 1

    os.makedirs(store_dir, exist_ok=True)
    with rnw.open(os.path.join(store_dir, "num_bins.json"), "wt") as f:
        f.write(
            json_numpy.dumps(
                {"ene": int(num_ene_bins), "dir": int(num_dir_bins)}
            )
        )

    for dir_bin in range(num_dir_bins):
        dir_bin_path = os.path.join(
            store_dir, DIRECTION_BIN_DIR_STR.format(dir_bin=dir_bin)
        )
        os.makedirs(dir_bin_path, exist_ok=True)

        for ene_bin in range(num_ene_bins):
            dir_ene_bin_path = os.path.join(
                dir_bin_path, ENERGY_BIN_DIR_STR.format(ene_bin=ene_bin)
            )
            os.makedirs(dir_ene_bin_path, exist_ok=True)

            page.write(
                path=os.path.join(dir_ene_bin_path, "cherenkov.rec"),
                page=page.init(size=0),
            )
            os.makedirs(
                os.path.join(dir_ene_bin_path, "cherenkov_stage"),
                exist_ok=True,
            )

            page.write(
                path=os.path.join(dir_ene_bin_path, "particle.rec"),
                page=page.init(size=0),
            )
            os.makedirs(
                os.path.join(dir_ene_bin_path, "particle_stage"), exist_ok=True
            )


class Store:
    def __init__(self, store_dir):
        self.store_dir = store_dir
        with open(os.path.join(self.store_dir, "num_bins.json"), "rt") as f:
            num_bins = json_numpy.loads(f.read())
        self.num_ene_bins = num_bins["ene"]
        self.num_dir_bins = num_bins["dir"]
        self.cache = {"cherenkov": {}, "particle": {}}

    def __contains__(self, dir_ene_bin):
        return self.contains_dir_ene_bin(dir_ene_bin=dir_ene_bin)

    def get_bin(self, dir_ene_bin, key):
        if not dir_ene_bin in self.cache[key]:
            self.cache[key][dir_ene_bin] = self.read_bin(
                dir_ene_bin=dir_ene_bin,
                key=key,
            )
        return self.cache[key][dir_ene_bin]

    def get_cherenkov_bin(self, dir_ene_bin):
        return self.get_bin(dir_ene_bin=dir_ene_bin, key="cherenkov")

    def __repr__(self):
        out = "{:s}(num bins: energy={:d}, direction={:d})".format(
            self.__class__.__name__,
            self.num_ene_bins,
            self.num_dir_bins,
        )
        return out

    def dir_ene_bin_path(self, dir_ene_bin):
        """
        Returns the path of a direction-energy-bin.
        """
        assert self.contains_dir_ene_bin(dir_ene_bin=dir_ene_bin)
        dir_bin, ene_bin = dir_ene_bin
        return os.path.join(
            self.store_dir,
            DIRECTION_BIN_DIR_STR.format(dir_bin=dir_bin),
            ENERGY_BIN_DIR_STR.format(ene_bin=ene_bin),
        )

    def contains_dir_ene_bin(self, dir_ene_bin):
        dir_bin, ene_bin = dir_ene_bin
        if dir_bin < 0 or dir_bin >= self.num_dir_bins:
            return False
        if ene_bin < 0 or ene_bin >= self.num_ene_bins:
            return False
        return True

    def _num_showers_in_file(self, dir_ene_bin, filename):
        return page.num_records_in_file(
            path=os.path.join(
                self.dir_ene_bin_path(dir_ene_bin=dir_ene_bin),
                filename,
            )
        )

    def population(self, key):
        pop = np.zeros(
            shape=(self.num_dir_bins, self.num_ene_bins),
            dtype=np.uint64,
        )
        for dir_bin in self.list_dir_bins():
            for ene_bin in self.list_ene_bins():
                pop[dir_bin, ene_bin] = self._num_showers_in_file(
                    dir_ene_bin=(dir_bin, ene_bin),
                    filename="{:s}.rec".format(key),
                )
        return pop

    def population_cherenkov(self):
        return self.population(key="cherenkov")

    def population_particle(self):
        return self.population(key="particle")

    def make_empty_stage(self, run_id):
        stage = {}
        stage["run_id"] = int(run_id)
        stage["records"] = {}
        for dir_ene_bin in self.list_dir_ene_bins():
            stage["records"][dir_ene_bin] = []
        return stage

    def add_cherenkov_to_stage(self, cherenkov_stage):
        self.add_to_stage(stage=cherenkov_stage, key="cherenkov")

    def add_particle_to_stage(self, particle_stage):
        self.add_to_stage(stage=particle_stage, key="particle")

    def add_to_stage(self, stage, key):
        for dir_ene_bin in self.list_dir_ene_bins():
            staged_records = stage["records"][dir_ene_bin]
            if len(staged_records) > 0:
                _dyn = dynamicsizerecarray.DynamicSizeRecarray(
                    dtype=page.dtype()
                )
                _dyn.append_records(staged_records)
                staged_showers = _dyn.to_recarray()

                stage_path = os.path.join(
                    self.dir_ene_bin_path(dir_ene_bin=dir_ene_bin),
                    "{:s}_stage".format(key),
                    "{:06d}.rec".format(stage["run_id"]),
                )
                page.write(path=stage_path, page=staged_showers)

    def commit_stage(self):
        num = {}
        for key in ["cherenkov", "particle"]:
            num[key] = 0
            for dir_ene_bin in self.list_dir_ene_bins():
                num[key] += self.commit_bin_stage(
                    dir_ene_bin=dir_ene_bin, key=key
                )
        return num

    def list_dir_ene_bins(self):
        dir_ene_bins = []
        for dir_bin in self.list_dir_bins():
            for ene_bin in self.list_ene_bins():
                dir_ene_bins.append((dir_bin, ene_bin))
        return dir_ene_bins

    def list_dir_bins(self):
        return np.arange(0, self.num_dir_bins)

    def list_ene_bins(self):
        return np.arange(0, self.num_ene_bins)

    def commit_bin_stage(self, dir_ene_bin, key):
        join = os.path.join
        dir_ene_bin_dir = self.dir_ene_bin_path(dir_ene_bin=dir_ene_bin)
        stage_dir = join(dir_ene_bin_dir, "{:s}_stage".format(key))
        existing_showers_path = join(dir_ene_bin_dir, "{:s}.rec".format(key))
        staged_showers_paths = glob.glob(join(stage_dir, "*.rec"))

        all_showers_dyn = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=page.dtype()
        )

        for staged_showers_path in staged_showers_paths:
            additional_showers = page.read(path=staged_showers_path)
            all_showers_dyn.append_recarray(additional_showers)
            os.remove(staged_showers_path)

        num_showers_in_stage = int(all_showers_dyn.size)

        if os.path.exists(existing_showers_path):
            existing_showers = page.read(path=existing_showers_path)
            all_showers_dyn.append_recarray(existing_showers)

        all_showers = all_showers_dyn.to_recarray()
        page.write(path=existing_showers_path, page=all_showers)
        return num_showers_in_stage

    def read_bin(self, dir_ene_bin, key):
        """
        Reads the 'key'-showers in 'dir_ene_bin' from the store and returns it.

        Parameters
        ----------
        dir_ene_bin : tuple(int, int)
            Direction- and energy-bin index.
        key : str
            Either 'cherenkov' or 'particle'.
        """
        dir_ene_bin_dir = self.dir_ene_bin_path(dir_ene_bin=dir_ene_bin)
        showers_path = os.path.join(dir_ene_bin_dir, "{:s}.rec".format(key))
        return page.read(path=showers_path)
