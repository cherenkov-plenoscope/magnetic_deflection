import os


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
            os.makedirs(os.path.join(d_e_dir, "cherenkov"), exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "cherenkov", "stage"), exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "particle"), exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "particle", "stage"), exist_ok=True)

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

    def num_cherenkov_statistics(self):
        pass




def shower_dtype():
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


def init_leaf():
    return np.core.records.recarray(
        shape=0,
        dtype=shower_dtype(),
    )


class DynamicSizeRecarray:
    def __init__(self, recarr):
        self.size = len(recarr)
        next_capacity = np.max([2, self.size])
        self.recarr = np.core.records.recarray(
            shape=next_capacity,
            dtype=recarr.dtype,
        )

    def capacity(self):
        return len(self.recarr)

    def to_recarray(self):
        out = np.core.records.recarray(
            shape=self.size,
            dtype=recarr.dtype,
        )
        out = self.recarr[0:self.size]
        return out

    def append_record(self, record):
        self._grow_if_needed(additional_size=1)

        for key in self.recarr.dtype.names:
            self.recarr[self.size][key] = record[key]

        self.size += 1

    def append_recarray(self, recarr):
        self._grow_if_needed(additional_size=len(recarr))

        self.recarr[self.size:] = recarr

        self.size += len(recarr)

    def _grow_if_needed(self, additional_size):
        assert additional_size >= 0
        current_capacity = self.capacity()
        required_size = self.size + additional_size

        if required_size > current_capacity:
            swp = copy.deepcopy(self.recarr)
            next_capacity = np.max([current_capacity * 2, required_size])
            self.recarr = np.core.records.recarray(
                shape=next_capacity,
                dtype=recarr.dtype,
            )
            self.recarr = copy.deepcopy(swp[0:self.size])
            del swp

    def __repr__(self):
        out = "{:s}(num bins: energy={:d}, direction={:d})".format(
            self.__class__.__name__,
            self.energy_num_bins,
            self.direction_num_bins,
        )
        return out

"""
run_id
event_id

cer-stats


def shower_dtype():
    return [
        "run": {"full": "u4": "compressed": None},
        "event": {"full": "u4": "compressed": None},
        "particle_azimuth_deg": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-181, 181]}},
        "particle_zenith_deg": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1, 91]}},
        "particle_energy_GeV": {"full": "f8": "compressed": {"dtype": "u2", "map": "log10", "map": [-3, 3]}},
        "cherenkov_num_photons": {"full": "f8": "compressed": {"dtype": "u2", "map": "log10", "map": [-1, 6]}},
        "cherenkov_num_bunches": {"full": "u8": "compressed": {"dtype": "u2", "map": "log10", "map": [-1, 6]}},
        "cherenkov_x_m": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1e5, 1e5]}},
        "cherenkov_y_m": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1e5, 1e5]}},
        "cherenkov_radius50_m": {"full": "f8": "compressed": {"dtype": "u2", "map": "log10", "map": [-1, 6]}},
        "cherenkov_cx_rad": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1.01, 1.01]}},
        "cherenkov_cy_rad": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1.01, 1.01]}},
        "cherenkov_angle50_rad": {"full": "f8": "compressed": {"dtype": "u2", "map": "1", "map": [-1.01, 1.01]}},
        "cherenkov_t_s": "f4",
        "cherenkov_t_std_s": "f4",
    ]


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def compress_zero_to_one(x, x_start, x_stop):
    assert np.all(x >= x_start)
    assert np.all(x < x_stop)
    assert x_start < x_stop
    return (x - x_start) / (x_stop - x_start)


def decompress_zero_to_one(kx, x_start, x_stop):
    assert np.all(kx >= 0)
    assert np.all(kx < 1)
    assert x_start < x_stop
    return (kx * (x_stop - x_start)) + x_start
"""