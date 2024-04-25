import numpy as np
import dynamicsizerecarray
import corsika_primary
import homogeneous_transformation
import spherical_coordinates
import atmospheric_cherenkov_response as acr


class SkyMapDummy:
    """
    A dummy for skymap.SkyMap() which is meant for testing.
    The population of skymap.SkyMap() takes large amounts of compute resources.
    To test functions which act on an skymap.SkyMap(), this dummy provides
    the central function: query_cherenkov_ball()
    """

    def __init__(self, site_key="chile", particle_key="electron"):
        self.config = {}
        self.config["particle"] = acr.particles.init(particle_key)
        self.config["site"] = acr.sites.init(site_key)
        self.config["binning"] = {}
        self.config["binning"]["energy"] = {
            "start": 0.25,
            "stop": 64.0,
            "num_bins": 10,
        }

    def draw(
        self,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_start_GeV,
        energy_stop_GeV,
        threshold_cherenkov_density_per_sr,
        solid_angle_sr,
        prng,
    ):
        random_seed = abs(
            hash(
                (
                    azimuth_rad,
                    zenith_rad,
                    half_angle_rad,
                    energy_start_GeV,
                    energy_stop_GeV,
                    threshold_cherenkov_density_per_sr,
                    solid_angle_sr,
                )
            )
        )
        prng = np.random.Generator(np.random.PCG64(random_seed))

        cer_cx, cer_cy, cer_cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
        )
        cer_cxcycz = np.array([cer_cx, cer_cy, cer_cz])

        deflection_homtra_start = _guess_deflection(
            energy_GeV=energy_start_GeV,
            prng=prng,
            random_angle_deg=1.0,
        )
        deflection_homtra_stop = _guess_deflection(
            energy_GeV=energy_stop_GeV,
            prng=prng,
            random_angle_deg=1.0,
        )
        prm_start_cxcycz = homogeneous_transformation.transform_orientation(
            t=deflection_homtra_start,
            d=cer_cxcycz,
        )
        prm_stop_cxcycz = homogeneous_transformation.transform_orientation(
            t=deflection_homtra_stop,
            d=cer_cxcycz,
        )
        prm_cxcycz = 0.5 * (prm_start_cxcycz + prm_stop_cxcycz)
        prm_cxcycz = prm_cxcycz / np.linalg.norm(prm_cxcycz)
        prm_az, prm_zd = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=prm_cxcycz[0],
            cy=prm_cxcycz[1],
            cz=prm_cxcycz[2],
        )
        res = {}
        if prm_zd < corsika_primary.MAX_ZENITH_DISTANCE_RAD:
            res["cutoff"] = False
            res["particle_azimuth_rad"] = prm_az
            res["particle_zenith_rad"] = prm_zd
            res["solid_angle_thrown_sr"] = prng.uniform(0.1, 0.35)
        else:
            res["cutoff"] = True

        dbg = {"method": "SkyMapDummy.draw()"}
        return res, dbg


def _guess_deflection(energy_GeV, random_angle_deg, prng):
    pure = _guess_pure_deflection(energy_GeV=energy_GeV)
    rand = _guess_random_rotation(prng=prng, magnitude_deg=random_angle_deg)
    return homogeneous_transformation.sequence(pure, rand)


def _guess_pure_deflection(energy_GeV):
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


def _guess_random_rotation(prng, magnitude_deg):
    m = magnitude_deg
    assert m >= 0.0
    civil_rotation = {
        "repr": "tait_bryan",
        "xyz_deg": [
            prng.uniform(-m, m),
            prng.uniform(-m, m),
            prng.uniform(-m, m),
        ],
    }
    return homogeneous_transformation.compile(
        {
            "pos": [0, 0, 0],
            "rot": civil_rotation,
        }
    )
