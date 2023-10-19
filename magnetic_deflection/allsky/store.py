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


def record_dtype():
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
