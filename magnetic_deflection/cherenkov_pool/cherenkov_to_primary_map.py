import svg_cartesian_plot as svgplt
import un_bound_histogram
import binning_utils
import spherical_coordinates
import spherical_histogram
import corsika_primary

import numpy as np


class CherenkovToPrimaryMap:
    def __init__(
        self,
        sky_bin_geometry,
        energy_bin_edges_GeV,
        altitude_bin_edges_m,
    ):
        assert binning_utils.is_strictly_monotonic_increasing(
            energy_bin_edges_GeV
        )
        assert np.all(energy_bin_edges_GeV > 0.0)

        assert binning_utils.is_strictly_monotonic_increasing(
            altitude_bin_edges_m
        )
        assert np.all(altitude_bin_edges_m >= 0.0)

        self.sky_bin_geometry = sky_bin_geometry
        self.energy_bin = binning_utils.Binning(bin_edges=energy_bin_edges_GeV)
        self.altitude_bin = binning_utils.Binning(
            bin_edges=altitude_bin_edges_m
        )

        self.map = np.zeros(
            shape=(
                len(self.sky_bin_geometry.faces),  # num cherenkov directions
                self.energy_bin["num"],
                self.altitude_bin["num"],
                len(self.sky_bin_geometry.faces),  # num primary directions
            ),
            dtype=np.uint8,
        )

        self.exposure = np.zeros(
            shape=(
                len(self.sky_bin_geometry.faces),  # num cherenkov directions
                self.energy_bin["num"],
                self.altitude_bin["num"],
            ),
            dtype=np.uint64,
        )

        self.overflow = []

    @classmethod
    def from_defaults(cls):
        return cls(
            sky_bin_geometry=spherical_histogram.geometry.HemisphereGeometry(
                num_vertices=511,
                max_zenith_distance_rad=np.deg2rad(89),
            ),
            energy_bin_edges_GeV=np.geomspace(2 ** (-2), 2 ** (6), 32 + 1),
            altitude_bin_edges_m=np.geomspace(2**10, 2**16, 3 + 1),
        )

    def __repr__(self):
        return "{:s}(sky {:d}, energy {:d}, altitude {:d})".format(
            self.__class__.__name__,
            len(self.sky_bin_geometry.faces),
            self.energy_bin["num"],
            self.altitude_bin["num"],
        )

    def assign(self, pool, cherenkov_sky_mask):
        assert cherenkov_sky_mask.shape[0] == len(self.sky_bin_geometry.faces)

        matching_bin = self.find_matching_bin(
            particle_cx=pool["particle_cx"],
            particle_cy=pool["particle_cy"],
            particle_energy_GeV=pool["particle_energy_GeV"],
            cherenkov_altitude_p50_m=pool["cherenkov_altitude_p50_m"],
        )

        if matching_bin["overflow"]:
            report = {}
            report["run"] = pool["run"]
            report["event"] = pool["event"]
            report["matching_bin"] = matching_bin
            self.overflow.append(report)
        else:
            m = matching_bin
            # exposure
            # --------
            self.exposure[m["sky"], m["ene"], m["alt"]] += 1

            # map
            # ---
            stage = self.map[m["sky"], m["ene"], m["alt"]].astype(np.uint64)
            stage[cherenkov_sky_mask] += 1
            assert np.all(stage <= np.iinfo(np.uint8).max)
            self.map[m["sky"], m["ene"], m["alt"]] = stage

    def find_matching_bin(
        self,
        particle_cx,
        particle_cy,
        particle_energy_GeV,
        cherenkov_altitude_p50_m,
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

        alt = -1 + np.digitize(
            x=cherenkov_altitude_p50_m, bins=self.altitude_bin["edges"]
        )
        alt_overflow = alt == self.altitude_bin["num"] or alt == -1

        return {
            "overflow": np.any([sky_overflow, ene_overflow, alt_overflow]),
            "sky_overflow": sky_overflow,
            "ene_overflow": ene_overflow,
            "alt_overflow": alt_overflow,
            "sky": sky,
            "ene": ene,
            "alt": alt,
        }
