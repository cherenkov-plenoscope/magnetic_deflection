import scipy
from .. import spherical_coordinates
from scipy import spatial
import binning_utils
import numpy as np


class Binning:
    def __init__(self, config):
        """
        A binning in directions and energy.
        The direction-binning is in a sphere.
        The energy binning is a geomspace.

        Compiles the Binning from the config-dict read from work_dir/config.
        """
        self.energy = binning_utils.Binning(
            _init_energy_bin_edges(
                start_GeV=config["energy"]["start_GeV"],
                stop_GeV=config["energy"]["stop_GeV"],
                num_bins=config["energy"]["num_bins"],
            )
        )
        centers = binning_utils.sphere.fibonacci_space(
            size=config["direction"]["num_bins"],
            max_zenith_distance_rad=np.deg2rad(
                config["direction"]["cherenkov_max_zenith_distance_deg"]
            ),
        )
        self.direction = scipy.spatial.cKDTree(data=centers)

    def query(self, azimuth_deg, zenith_deg, energy_GeV):
        """
        Finds the closest bin for a given direction and energy.
        The direction is given in (azimuth_deg, zenith_deg).

        Parameters
        ----------
        azimuth_deg : float
            Azimuth angle to find the closest match for in DEG.
        zenith_deg : float
            Zenith angle to find the closest match for in DEG.
        energy_GeV : float
            Energy to find the closest match for in GeV.

        Retruns
        -------
        (p_angle_deg, e_distance_GeV), (pbin, ebin)

        p_angle_deg : float
            Angle to the closest direction-bin-center.
        e_distance_GeV : float
            Absolute energy-distance to the closest bin-edge in energy.
        pbin : int
            Index of the closest direction-bin.
        ebin : int
            Index of the closest energy-bin.
        """
        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        pointing = np.array([cx, cy, cz])
        p_angle_rad, pbin = self.direction.query(pointing)
        ebin = np.digitize(energy_GeV, self.energy["edges"]) - 1

        ee = self.energy["edges"] / energy_GeV
        ee[ee < 1] = 1 / ee[ee < 1]
        eclose = np.argmin(ee)

        e_distance_GeV = np.abs(self.energy["edges"][eclose] - energy_GeV)

        p_angle_deg = np.rad2deg(p_angle_rad)
        return (p_angle_deg, e_distance_GeV), (pbin, ebin)

    def query_ball(
        self,
        azimuth_deg,
        zenith_deg,
        half_angle_deg,
        energy_start_GeV,
        energy_stop_GeV,
    ):
        """
        Finds the bins within a given direction-cone and energy-range.

        Parameters
        ----------
        azimuth_deg : float
            Direction's azimuth in DEG.
        zenith_deg : float
            Direction's zenith in DEG.
        half_angle_deg : float
            Cone's half angle (on the sky-dome) which encircles the direction.
            All direction-bins within this cone will be returned.
        energy_start_GeV : float
            Start of energy-range in GeV.
        energy_stop_GeV : float
            Stop of energy-range in GeV.

        Returns
        -------
        (pp, ee)

        pp : np.array
            Indices of direction-bins which are within the query's radius.
        ee : np.array
            Indices of energy-bins which are within the query's radius.
        """
        assert energy_start_GeV > 0
        assert energy_stop_GeV > 0
        assert energy_stop_GeV > energy_start_GeV
        assert half_angle_deg >= 0

        cx, cy, cz = spherical_coordinates._az_zd_to_cx_cy_cz(
            azimuth_deg=azimuth_deg, zenith_deg=zenith_deg
        )
        pointing = np.array([cx, cy, cz])
        pp = self.direction.query_ball_point(
            x=pointing, r=np.deg2rad(half_angle_deg)
        )
        pp = np.array(pp)

        ebin_start = np.digitize(energy_start_GeV, self.energy["edges"]) - 1
        ebin_stop = np.digitize(energy_stop_GeV, self.energy["edges"]) - 1

        ee = set()

        if ebin_start >= 0 and ebin_start < self.energy["num"]:
            ee.add(ebin_start)
        if ebin_stop >= 0 and ebin_stop < self.energy["num"]:
            ee.add(ebin_stop)

        ee = np.array(list(ee))
        if len(ee) == 2:
            ee = np.arange(min(ee), max(ee) + 1)

        return (pp, ee)


def _init_energy_bin_edges(start_GeV, stop_GeV, num_bins):
    return np.geomspace(start_GeV, stop_GeV, num_bins + 1)