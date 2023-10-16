import numpy as np
import binning_utils
import os
import json_utils
import json_utils
import rename_after_writing as rnw
import scipy
import gzip
import pandas
from scipy import spatial
from .. import spherical_coordinates


def init_dome(
    dome_dir,
    direction_max_zenith_distance_deg=60,
    direction_num_bins=256,
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=32,
):
    join = os.path.join
    energy_bin_edges = init_energy_bin_edges(
        start_GeV=energy_start_GeV,
        stop_GeV=energy_stop_GeV,
        num_bins=energy_num_bins,
    )

    direction_bin_centers = init_direction_bin_centers(
        max_zenith_distance_deg=direction_max_zenith_distance_deg,
        num_bins=direction_num_bins,
    )

    os.makedirs(dome_dir, exist_ok=True)
    bin_dir = join(dome_dir, "binning")
    os.makedirs(bin_dir, exist_ok=True)

    with rnw.open(join(bin_dir, "energy_bin.json"), "wt") as f:
        f.write(
            json_utils.dumps(binning_utils.Binning(energy_bin_edges), indent=4)
        )

    with rnw.open(join(bin_dir, "direction_bin.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                {
                    "centers": direction_bin_centers,
                    "max_zenith_distance_deg": direction_max_zenith_distance_deg,
                },
                indent=4,
            )
        )

    num_direction_centers = direction_bin_centers.shape[0]

    for poi in range(num_direction_centers):
        poi_str = "{:06d}".format(poi)
        for ene in range(energy_num_bins):
            ene_str = "{:06d}".format(ene)
            os.makedirs(join(dome_dir, poi_str, ene_str), exist_ok=True)
            os.makedirs(
                join(dome_dir, poi_str, ene_str, "stage"), exist_ok=True
            )
            os.makedirs(
                join(dome_dir, poi_str, ene_str, "cherenkov"), exist_ok=True
            )
            os.makedirs(
                join(dome_dir, poi_str, ene_str, "primary"), exist_ok=True
            )


def init_energy_bin_edges(start_GeV, stop_GeV, num_bins):
    return np.geomspace(start_GeV, stop_GeV, num_bins + 1)


def init_direction_bin_centers(max_zenith_distance_deg, num_bins):
    return binning_utils.sphere.fibonacci_space(
        size=num_bins,
        max_zenith_distance_rad=np.deg2rad(max_zenith_distance_deg),
    )


def dome_read_binning(dome_dir):
    raw = json_utils.tree.read(os.path.join(dome_dir, "binning"))
    pointing_bin_centers = raw["direction_bin"].pop("centers")
    raw["direction_bin"]["centers_tree"] = scipy.spatial.cKDTree(
        data=pointing_bin_centers,
    )
    return raw


def dome_query_bin(dome_binning, azimuth_deg, zenith_deg, energy_GeV):
    dbin = dome_binning
    cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
        azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
    )
    pointing = np.array([cx, cy, cz])
    p_angle_rad, pbin = dbin["direction_bin"]["centers_tree"].query(pointing)
    ebin = np.digitize(energy_GeV, dbin["energy_bin"]["edges"]) - 1

    ee = dbin["energy_bin"]["edges"] / energy_GeV
    ee[ee < 1] = 1 / ee[ee < 1]
    eclose = np.argmin(ee)

    e_distance_GeV = np.abs(dbin["energy_bin"]["edges"][eclose] - energy_GeV)

    p_angle_deg = np.rad2deg(p_angle_rad)
    return (p_angle_deg, e_distance_GeV), (pbin, ebin)


def dome_query_bin_ball(
    dome_binning,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    energy_start_GeV,
    energy_stop_GeV,
):
    dbin = dome_binning
    cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
        azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
    )
    pointing = np.array([cx, cy, cz])
    pp = dbin["direction_bin"]["centers_tree"].query_ball_point(
        x=pointing, r=np.deg2rad(half_angle_deg)
    )
    pp = np.array(pp)

    ebin_start = np.digitize(energy_start_GeV, dbin["energy_bin"]["edges"]) - 1
    ebin_stop = np.digitize(energy_stop_GeV, dbin["energy_bin"]["edges"]) - 1

    ee = set()

    if ebin_start >= 0 and ebin_start < dbin["energy_bin"]["num"]:
        ee.add(ebin_start)
    if ebin_stop >= 0 and ebin_stop < dbin["energy_bin"]["num"]:
        ee.add(ebin_stop)

    ee = np.array(list(ee))
    if len(ee) == 2:
        ee = np.arange(min(ee), max(ee) + 1)

    return (pp, ee)


def make_bins(pointing_bins, energy_bins):
    out = set()
    for pointing_bin in pointing_bins:
        for energy_bin in energy_bins:
            bb = (pointing_bin, energy_bin)
            out.add(bb)
    return out


class Dome:
    def __init__(self, dome_dir, max_cache_size=1000):
        self.dome_dir = dome_dir
        self.binning = dome_read_binning(dome_dir=self.dome_dir)
        self.cache = {}
        self.max_cache_size = max_cache_size

    def query(self, azimuth_deg, zenith_deg, energy_GeV):
        return dome_query_bin(
            dome_binning=self.binning,
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            energy_GeV=energy_GeV,
        )

    def update_cache(self, required_bins):
        rec = []
        for required_bin in required_bins:
            if required_bin not in self.cache:
                bin_dir = make_domebin_dir(required_bin)
                bin_path = os.path.join(
                    self.dome_dir, bin_dir, "cherenkov.jsonl"
                )
                if not os.path.isfile(bin_path):
                    print("No bin : {:s}".format(bin_path))
                    continue
                else:
                    rec.append(required_bin)

                bin_content = json_utils.lines.read(bin_path)
                bin_content = pandas.DataFrame(bin_content).to_records(
                    index=False
                )
                self.cache[required_bin] = bin_content
            else:
                rec.append(required_bin)
        return rec

        """
        age_of_not_required = {}

        # list bins which can leave cache
        num_can_leave = 0
        for key in self.cache:
            if key not in required_bins:
                age = self.cache[key]["age"]
                if age not in age_of_not_required:
                    age_of_not_required[age] = [key]
                else:
                    age_of_not_required[age].append(key)
                num_can_leave += 1

        ages = np.flip(np.sort(list(age_of_not_required.keys())))
        bins_to_leave_cache = []

        required_size = len(required_bins)
        cache_size = len(self.cache)

        num_bins_to_leave = self.max_cache_size
        """

    def query_ball(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_start_GeV,
        energy_stop_GeV,
    ):
        pointing_bins, energy_bins = dome_query_bin_ball(
            dome_binning=self.binning,
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=half_angle_deg,
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
        )

        required_bins = make_bins(
            pointing_bins=pointing_bins,
            energy_bins=energy_bins,
        )

        required_bins = self.update_cache(required_bins=required_bins)

        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
        )
        cxcycz = np.array([cx, cy, cz])

        matches = []
        for rbin in required_bins:
            energy_mask = np.logical_and(
                self.cache[rbin]["particle_energy_GeV"] >= energy_start_GeV,
                self.cache[rbin]["particle_energy_GeV"] < energy_stop_GeV,
            )
            cherenkov_cz_rad = make_cherenkov_cz_from_leaf(self.cache[rbin])
            cer = np.zeros(shape=(cherenkov_cz_rad.shape[0], 3))
            cer[:, 0] = self.cache[rbin]["cherenkov_cx_rad"]
            cer[:, 1] = self.cache[rbin]["cherenkov_cy_rad"]
            cer[:, 2] = cherenkov_cz_rad

            delta = cer - cxcycz
            delta2_rad2 = np.sum(delta**2, axis=1)
            assert delta2_rad2.shape[0] == cherenkov_cz_rad.shape[0]

            direction_mask = delta2_rad2 < np.deg2rad(half_angle_deg) ** 2
            mask = np.logical_and(direction_mask, energy_mask)

            print(
                "rbin",
                rbin,
                "E",
                np.sum(energy_mask),
                "D",
                np.sum(direction_mask),
                "ED",
                np.sum(mask),
            )

            part = self.cache[rbin][mask]
            matches.append(part)
        return np.hstack(matches)

    def list_domebins(self):
        num_direction_bins = self.binning["direction_bin"][
            "centers_tree"
        ].data.shape[0]
        domebins = []
        for dbin in range(num_direction_bins):
            for ebin in range(self.binning["energy_bin"]["num"]):
                domebin = (dbin, ebin)
                domebins.append(domebin)
        return domebins


def make_cherenkov_cz_from_leaf(leaf):
    cx = leaf["cherenkov_cx_rad"]
    cy = leaf["cherenkov_cy_rad"]
    cz = np.sqrt(1.0 - cx**2 - cy**2)
    return cz


def make_domebin_dir(domebin):
    dirbin, ebin = domebin
    return os.path.join("{:06d}".format(dirbin), "{:06d}".format(ebin))


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


def compress_leaf(leaf, energy_start, energy_stop):
    energy_f = compress_zero_to_one(
        x=leaf["energy_GeV"], x_start=energy_start, x_stop=energy_stop
    )

    primary_cx_f = compress_zero_to_one(
        x=leaf["primary_cx_rad"], x_start=-1, x_stop=1
    )
    primary_cy_f = compress_zero_to_one(
        x=leaf["primary_cy_rad"], x_start=-1, x_stop=1
    )

    cherenkov_cx_f = compress_zero_to_one(
        x=leaf["cherenkov_cx_rad"], x_start=-1, x_stop=1
    )
    cherenkov_cy_f = compress_zero_to_one(
        x=leaf["cherenkov_cy_rad"], x_start=-1, x_stop=1
    )

    m16 = np.iinfo(np.uint16).max
    leaf = np.zeros(shape=(num, 5), dtype=np.uint16)
    leaf[:, 0] = np.array(cherenkov_cx_f * m16, dtype=np.uint16)
    leaf[:, 1] = np.array(cherenkov_cy_f * m16, dtype=np.uint16)
    leaf[:, 2] = np.array(primary_cx_f * m16, dtype=np.uint16)
    leaf[:, 3] = np.array(primary_cy_f * m16, dtype=np.uint16)
    leaf[:, 4] = np.array(energy_f * m16, dtype=np.uint16)
    return leaf


def decompress_leaf(comp_leaf):
    m16 = np.float64(np.iinfo(np.uint16).max)
    out = {}
    cherenkov_cx_f = np.array(comp_leaf[:, 0], dtype=np.float64) / m16
    out["cherenkov_cx_rad"] = decompress_zero_to_one(
        kx=cherenkov_cx_f, x_start=-1, x_stop=1
    )

    cherenkov_cy_f = np.array(comp_leaf[:, 1], dtype=np.float64) / m16
    out["cherenkov_cy_rad"] = decompress_zero_to_one(
        kx=cherenkov_cy_f, x_start=-1, x_stop=1
    )

    primary_cx_f = np.array(comp_leaf[:, 2], dtype=np.float64) / m16
    out["primary_cx_rad"] = decompress_zero_to_one(
        kx=primary_cx_f, x_start=-1, x_stop=1
    )

    primary_cy_f = np.array(comp_leaf[:, 3], dtype=np.float64) / m16
    out["primary_cy_rad"] = decompress_zero_to_one(
        kx=primary_cy_f, x_start=-1, x_stop=1
    )

    energy_f = np.array(comp_leaf[:, 4], dtype=np.float64) / m16
    out["energy_GeV"] = decompress_zero_to_one(
        kx=energy_f, x_start=-1, x_stop=1
    )

    return out
