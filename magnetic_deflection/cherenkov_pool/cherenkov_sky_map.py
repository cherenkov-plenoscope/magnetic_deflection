import binning_utils
import spherical_histogram
import corsika_primary

import numpy as np


def report_dtype():
    return [
        ("map_sky_bin", "i2"),
        ("map_energy_bin", "i2"),
    ]


class CherenkovSkyMap:
    def __init__(
        self,
        sky_bin_geometry,
        energy_bin_edges_GeV,
    ):
        assert binning_utils.is_strictly_monotonic_increasing(
            energy_bin_edges_GeV
        )
        assert np.all(energy_bin_edges_GeV > 0.0)

        self.sky_bin_geometry = sky_bin_geometry
        self.energy_bin = binning_utils.Binning(bin_edges=energy_bin_edges_GeV)

        self.primary_to_cherenkov = np.zeros(
            shape=(
                self.energy_bin["num"],
                self.num_sky_bins(),  # primary directions
                self.num_sky_bins(),  # cherenkov directions
            ),
            dtype=np.float32,
        )

        self.exposure = np.zeros(
            shape=(
                self.energy_bin["num"],
                self.num_sky_bins(),  # num cherenkov directions
            ),
            dtype=np.uint64,
        )

        self.overflow = 0

    def num_sky_bins(self):
        return len(self.sky_bin_geometry.faces)

    @classmethod
    def from_defaults(cls):
        return cls(
            sky_bin_geometry=spherical_histogram.geometry.HemisphereGeometry(
                num_vertices=511,
                max_zenith_distance_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
            ),
            energy_bin_edges_GeV=np.geomspace(2 ** (-2), 2 ** (6), 32 + 1),
        )

    def __repr__(self):
        return "{:s}(sky {:d}, energy {:d})".format(
            self.__class__.__name__,
            len(self.sky_bin_geometry.faces),
            self.energy_bin["num"],
        )

    def assign(
        self,
        particle_cx,
        particle_cy,
        particle_energy_GeV,
        cherenkov_sky_intensity,
    ):
        assert cherenkov_sky_intensity.shape[0] == self.num_sky_bins()

        matching_bin = self.find_matching_bin(
            particle_cx=particle_cx,
            particle_cy=particle_cy,
            particle_energy_GeV=particle_energy_GeV,
        )

        if matching_bin["overflow"]:
            self.overflow += 1
        else:
            # exposure
            # --------
            self.exposure[matching_bin["ene"], matching_bin["sky"]] += 1

            # primary_to_cherenkov
            # --------------------
            self.primary_to_cherenkov[
                matching_bin["ene"], matching_bin["sky"]
            ] += cherenkov_sky_intensity

        return {
            "map_sky_bin": matching_bin["sky"],
            "map_energy_bin": matching_bin["ene"],
        }

    def find_matching_bin(
        self,
        particle_cx,
        particle_cy,
        particle_energy_GeV,
    ):
        sky = self.sky_bin_geometry.query_cx_cy(
            cx=particle_cx,
            cy=particle_cy,
        )
        sky_overflow = sky == -1

        ene = -1 + np.digitize(
            x=particle_energy_GeV, bins=self.energy_bin["edges"]
        )
        ene_overflow = ene == self.energy_bin["num"] or ene == -1

        return {
            "overflow": np.any([sky_overflow, ene_overflow]),
            "sky_overflow": sky_overflow,
            "ene_overflow": ene_overflow,
            "sky": sky,
            "ene": ene,
        }
