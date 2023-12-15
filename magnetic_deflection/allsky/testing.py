import numpy as np
import dynamicsizerecarray
import corsika_primary
import homogeneous_transformation
import spherical_coordinates
import atmospheric_cherenkov_response as acr

from . import store


class AllSkyDummy:
    """
    A dummy for allsky.AllSky() which is meant for testing.
    The population of allsky.AllSky() takes large amounts of compute resources.
    To test functions which act on an allsky.AllSky(), this dummy provides
    the central function: query_cherenkov_ball()
    """

    def __init__(self):
        self.cache_dtype = store.minimal_cache_dtype()
        self.config = {}
        self.config["particle"] = acr.particles.init("electron")
        self.config["site"] = acr.sites.init("chile")
        self.config["binning"] = {}
        self.config["binning"]["direction"] = {}
        self.config["binning"]["direction"][
            "particle_max_zenith_distance_rad"
        ] = corsika_primary.MAX_ZENITH_RAD
        self.config["binning"]["direction"]["num_bins"] = 8
        self.config["binning"]["energy"] = {
            "start_GeV": 0.3981071705534972,
            "stop_GeV": 64.0,
            "num_bins": 3,
        }

    def _guess_deflection(self, energy_GeV):
        angle_deg = 11.0 / (energy_GeV**1.5)
        deflection_civil_rotation = {
            "repr": "axis_angle",
            "axis": [1, 1, 0],
            "angle_deg": angle_deg,
        }
        return homogeneous_transformation.compile(
            {
                "pos": [0, 0, 0],
                "rot": deflection_civil_rotation,
            }
        )

    def _draw_cherenkov_median_cx_cy(
        self, azimuth_rad, zenith_rad, half_angle_rad, prng
    ):
        (
            cer_az_rad,
            cer_zd_rad,
        ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=half_angle_rad,
            max_zenith_rad=np.pi,
            max_iterations=1000 * 1000,
        )

        return spherical_coordinates.az_zd_to_cx_cy(
            azimuth_rad=cer_az_rad, zenith_rad=cer_zd_rad
        )

    def query_cherenkov_ball(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_GeV,
        energy_factor,
        min_num_cherenkov_photons=1e3,
        weights=False,
    ):
        random_seed = abs(
            hash(
                (
                    azimuth_rad,
                    zenith_rad,
                    half_angle_rad,
                    energy_GeV,
                    energy_factor,
                    min_num_cherenkov_photons,
                    weights,
                )
            )
        )
        prng = np.random.Generator(np.random.PCG64(random_seed))

        matches = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=self.cache_dtype
        )
        num = int(prng.uniform(low=100, high=200))

        energy_start_GeV = energy_GeV * (1 - energy_factor)
        energy_stop_GeV = energy_GeV * (1 + energy_factor)

        energies_GeV = corsika_primary.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=energy_start_GeV,
            upper_limit=energy_stop_GeV,
            power_slope=-2.0,
            num_samples=num,
        )

        photon_gain_per_GeV = 500

        for i in range(num):
            rec = {}
            rec["particle_energy_GeV"] = energies_GeV[i]

            deflection_homtra = self._guess_deflection(
                energy_GeV=rec["particle_energy_GeV"]
            )

            rec["cherenkov_num_photons"] = (
                photon_gain_per_GeV
                * energies_GeV[i]
                * prng.uniform(low=0.5, high=2.0)
            )

            (
                rec["cherenkov_cx_rad"],
                rec["cherenkov_cy_rad"],
            ) = self._draw_cherenkov_median_cx_cy(
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                half_angle_rad=half_angle_rad,
                prng=prng,
            )

            cer_cxcycz = np.array(
                [
                    rec["cherenkov_cx_rad"],
                    rec["cherenkov_cy_rad"],
                    spherical_coordinates.restore_cz(
                        rec["cherenkov_cx_rad"],
                        rec["cherenkov_cy_rad"],
                    ),
                ]
            )
            prm_cxcycz = homogeneous_transformation.transform_orientation(
                t=deflection_homtra,
                d=cer_cxcycz,
            )
            rec["particle_cx_rad"] = prm_cxcycz[0]
            rec["particle_cy_rad"] = prm_cxcycz[1]

            prm_az, prm_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
                cx=prm_cxcycz[0],
                cy=prm_cxcycz[1],
                cz=prm_cxcycz[2],
            )

            if prm_zd < corsika_primary.MAX_ZENITH_RAD:
                matches.append_record(rec)

        return matches.to_recarray()
