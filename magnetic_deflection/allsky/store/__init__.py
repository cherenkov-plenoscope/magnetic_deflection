import os
import uuid
import glob
import numpy as np
import json_numpy
import rename_after_writing as rnw
import dynamicsizerecarray
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


def minimal_cache_dtype():
    """
    The minimal subset of page.dtype().
    """
    return [
        ("particle_cx_rad", "f4"),
        ("particle_cy_rad", "f4"),
        ("particle_energy_GeV", "f4"),
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_cx_rad", "f4"),
        ("cherenkov_cy_rad", "f4"),
    ]


class Store:
    def __init__(self, store_dir, cache_dtype=minimal_cache_dtype()):
        self.store_dir = store_dir
        with open(os.path.join(self.store_dir, "num_bins.json"), "rt") as f:
            num_bins = json_numpy.loads(f.read())
        self.num_ene_bins = num_bins["ene"]
        self.num_dir_bins = num_bins["dir"]
        self.cache = {}
        self.cache_dtype = cache_dtype

    def __contains__(self, dir_ene_bin):
        return self.contains_dir_ene_bin(dir_ene_bin=dir_ene_bin)

    def __getitem__(self, dir_ene_bin):
        if not dir_ene_bin in self.cache:
            bin_content_rec = self.read_bin(
                dir_ene_bin=dir_ene_bin,
                key="cherenkov",
            )
            self.cache[dir_ene_bin] = recarray_keep_only(
                rec=bin_content_rec,
                dtype=self.cache_dtype,
            )
        return self.cache[dir_ene_bin]

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
        return make_dir_ene_bin_path(
            store_dir=self.store_dir, dir_ene_bin=dir_ene_bin
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

    def commit_stage(self, pool=None):
        jobs = self._commit_stage_make_jobs()
        if pool is None:
            _nums = map(_commit_stage_run_job, jobs)
        else:
            _nums = pool.map(_commit_stage_run_job, jobs)
        return sum(_nums)

    def _commit_stage_make_jobs(self):
        jobs = []
        for key in ["cherenkov", "particle"]:
            for dir_ene_bin in self.list_dir_ene_bins():
                job = {}
                job["store_dir"] = str(self.store_dir)
                job["key"] = str(key)
                job["dir_ene_bin"] = dir_ene_bin
                jobs.append(job)
        return jobs

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

    def export_csv(self, path, fraction=1.0, seed=1337):
        """
        Dumps all thrown showers into a comma seperated value file in 'path'.

        Parameters
        ----------
        path : str
            Path of the written file.
        fraction : float
            Export only this fraction of showers into the tables.
        seed : int
        """
        assert 1.0 >= fraction > 0.0
        key = "particle"  # All thrown showers.

        prng = np.random.Generator(np.random.PCG64(seed))
        page_dtype = page.dtype()

        with rnw.open(path, "wt") as f:
            for i in range(len(page_dtype)):
                column_name = page_dtype[i][0]
                f.write(column_name)
                if (i + 1) < len(page_dtype):
                    f.write(",")
            f.write("\n")

            dir_ene_bins = self.list_dir_ene_bins()

            for j, dir_ene_bin in enumerate(dir_ene_bins):
                print(dir_ene_bin, (j + 1), "of", len(dir_ene_bins))

                showers = self.read_bin(dir_ene_bin=dir_ene_bin, key=key)
                mask = prng.uniform(low=0.0, high=1.0, size=len(showers))
                mask = mask < fraction
                showers = showers[mask]

                for ee in range(len(showers)):
                    for ii in range(len(page_dtype)):
                        column_name = page_dtype[ii][0]
                        column_dtype = page_dtype[ii][1]
                        val = showers[column_name][ee]

                        if "u" in column_dtype or "i" in column_dtype:
                            f.write("{:d}".format(val))
                        else:
                            f.write("{:e}".format(val))
                        if (ii + 1) < len(page_dtype):
                            f.write(",")
                    f.write("\n")


def recarray_keep_only(rec, dtype):
    out = np.core.records.recarray(
        shape=len(rec),
        dtype=dtype,
    )
    for dt in dtype:
        name = dt[0]
        out[name] = rec[name]
    return out


def _commit_stage_run_job(job):
    return _run_commit_bin_stage(**job)


def _run_commit_bin_stage(store_dir, dir_ene_bin, key):
    join = os.path.join
    dir_ene_bin_dir = make_dir_ene_bin_path(
        store_dir=store_dir, dir_ene_bin=dir_ene_bin
    )
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

    num_showers_in_stage = int(len(all_showers_dyn))

    if os.path.exists(existing_showers_path):
        existing_showers = page.read(path=existing_showers_path)
        all_showers_dyn.append_recarray(existing_showers)

    all_showers = all_showers_dyn.to_recarray()
    page.write(path=existing_showers_path, page=all_showers)
    return num_showers_in_stage


def make_dir_ene_bin_path(store_dir, dir_ene_bin):
    dir_bin, ene_bin = dir_ene_bin
    return os.path.join(
        store_dir,
        DIRECTION_BIN_DIR_STR.format(dir_bin=dir_bin),
        ENERGY_BIN_DIR_STR.format(ene_bin=ene_bin),
    )
