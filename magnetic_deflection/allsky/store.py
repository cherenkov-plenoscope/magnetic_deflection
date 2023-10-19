import os


DIRECTION_BIN_DIR_STR = "{direction_bin:06d}"
ENERGY_BIN_DIR_STR = "{energy_bin:06d}"


def init(storage_dir, energy_num_bins, direction_num_bins):
    assert direction_num_bins >= 1
    assert energy_num_bins >= 1

    os.makedirs(storage_dir, exist_ok=True)

    for d_bin in range(direction_num_bins):
        d_dir = os.path.join(
            storage_dir, DIRECTION_BIN_DIR_STR.format(direction_bin=d_bin)
        )
        os.makedirs(d_dir, exist_ok=True)

        for e_bin in range(energy_num_bins):
            d_e_dir = os.path.join(
                d_dir, ENERGY_BIN_DIR_STR.format(energy_bin=e_bin)
            )
            os.makedirs(d_e_dir, exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "cherenkov"), exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "particle"), exist_ok=True)
            os.makedirs(os.path.join(d_e_dir, "stage"), exist_ok=True)


class Storage:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
