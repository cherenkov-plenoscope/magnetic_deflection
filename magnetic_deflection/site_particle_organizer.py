from .version import __version__
from . import allsky
from . import cherenkov_pool
from . import common_settings_for_plotting
from . import utils
from . import skymap

import atmospheric_cherenkov_response
import os
import numpy as np
import json_numpy
import rename_after_writing as rnw
import corsika_primary
import subprocess
import binning_utils


def make_example_args_for_init():
    energy_start_GeV = binning_utils.power10.lower_bin_edge(
        decade=-1, bin=2, num_bins_per_decade=5
    )
    energy_stop_GeV = binning_utils.power10.lower_bin_edge(
        decade=1, bin=4, num_bins_per_decade=5
    )
    site_keys = atmospheric_cherenkov_response.sites.keys()
    particle_keys = atmospheric_cherenkov_response.particles.keys()
    ENERGY_POWER_SLOPE = -1.5

    out = {
        "site_keys": site_keys,
        "particle_keys": particle_keys,
        "energy_start_GeV": energy_start_GeV,
        "energy_stop_GeV": energy_stop_GeV,
        "energy_num_bins": 32,
        "energy_power_slope": ENERGY_POWER_SLOPE,
    }
    out.update(
        guess_sky_faces_sky_vertices_and_groun_bin_area(
            field_of_view_half_angle_rad=np.deg2rad(3.25),
            mirror_diameter_m=71.0,
        )
    )


def guess_sky_faces_sky_vertices_and_groun_bin_area(
    field_of_view_half_angle_rad,
    mirror_diameter_m,
):
    OVERHEAD = 2.0
    sky_vertices, sky_faces = skymap._guess_sky_vertices_and_faces(
        fov_half_angle_rad=field_of_view_half_angle_rad,
        num_faces_in_fov=OVERHEAD,
        max_zenith_distance_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
    )
    ground_bin_area_m2 = skymap._guess_ground_bin_area_m2(
        mirror_diameter_m=mirror_diameter_m,
        num_bins_in_mirror=OVERHEAD,
    )
    return {
        "sky_faces": sky_faces,
        "sky_vertices": sky_vertices,
        "ground_bin_area_m2": ground_bin_area_m2,
    }


def init(
    work_dir,
    site_keys,
    particle_keys,
    energy_start_GeV,
    energy_stop_GeV,
    energy_num_bins,
    energy_power_slope,
    sky_faces,
    sky_vertices,
    ground_bin_area_m2,
):
    """
    Creates tables listing the magnetic deflection of atmospheric showers
    induced by particles. The tables cover the all the sky (see
    magnetic_deflection.allsky).

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
    energy_start_GeV : float
        Minimium energy, except the underlying simulation CORSIKA can not go
        this low for a specific particle. In this case the energy_start_GeV
        is set to the particle's minimal energy which is safe to simulate
        with CORSIKA.
    energy_stop_GeV : float
        Maximum energy of particles to simulate and populate the tables with.
        Showers induced by particles above this energy are considered to show
        no significant deflection in earth's magnetic field.
    site_keys : list [str]
        The keys (names) of the sites to be simulated.
        See atmospheric_cherenkov_response.sites package.
    particle_keys : list [str]
        The keys (names) of the particles to be simulated.
        See atmospheric_cherenkov_response.particles package.
    sky_faces : array_like, shape = (NUM_FACES, 3)

    sky_vertices : array_like, shape = (NUM_VERTICES, 3)

    example
    -------
    when site_keys is [namibia, chile] and
    particle_keys is [electron, proton, gamma], the work_dir
    will look like this:

    |-> work_dir
            |-> namibia
            |       |-> electron  <- each of these directories is an AllSky.
            |       |-> proton
            |       |-> gamma
            |
            |-> chile
                    |-> electron
                    |-> proton
                    |-> gamma

    """

    os.makedirs(work_dir, exist_ok=True)

    for sk in site_keys:
        sk_dir = os.path.join(work_dir, sk)
        for pk in particle_keys:
            sk_pk_dir = os.path.join(sk_dir, pk)

            if not os.path.exists(sk_pk_dir):
                particle = atmospheric_cherenkov_response.particles.init(pk)

                # Different particles have different minimal energies.
                # E.g. the proton needs much more energy to emitt Cherenkov
                # light than the electron.

                if particle["corsika"]["min_energy_GeV"]:
                    this_particle_energy_start_GeV = max(
                        [
                            particle["corsika"]["min_energy_GeV"],
                            energy_start_GeV,
                        ]
                    )
                else:
                    this_particle_energy_start_GeV = energy_start_GeV

                this_particle_energy_bin_edges_GeV = binning_utils.power.space(
                    start=this_particle_energy_start_GeV,
                    stop=energy_stop_GeV,
                    power_slope=energy_power_slope,
                    size=energy_num_bins + 1,
                )
                skymap.init(
                    work_dir=sk_pk_dir,
                    particle_key=pk,
                    site_key=sk,
                    energy_bin_edges_GeV=this_particle_energy_bin_edges_GeV,
                    energy_power_slope=energy_power_slope,
                    sky_vertices=sky_vertices,
                    sky_faces=sky_faces,
                    ground_bin_area_m2=ground_bin_area_m2,
                )


class SiteParticleOrganizer:
    def __init__(self, work_dir):
        self.work_dir = os.path.abspath(work_dir)
        self.site_keys, self.particle_keys = find_site_and_particle_keys(
            work_dir=self.work_dir
        )

    def population(self):
        return num_showers(work_dir=self.work_dir)

    def __repr__(self):
        out = "{:s}(work_dir='{:s}')".format(
            self.__class__.__name__, self.work_dir
        )
        return out


def find_site_and_particle_keys(work_dir):
    """
    Finds the site-keys and particle-keys which all sites have in common.

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.

    Returns
    -------
    (site_keys, particle_keys) : (list, list)
    """
    tree = utils._sniff_site_and_particle_keys(work_dir=work_dir)
    return utils._get_common_sites_and_particles(tree=tree)


def run(
    work_dir,
    pool,
    num_runs=192,
    num_showers_per_run=1280,
    num_showers_target=2 * 1000 * 1000,
):
    """
    Increase the population of showers in each table (site,particle)
    combination.

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
    pool : e.g. multiprocessing.Pool
        Needs to have a map() function. This is the thread or job pool used
        in the parallel production. When pool is builtins, builtins.map is
        used for serial processing.
    num_runs : int
        Add this many runs of showers. (A run is a production run of showers).
    num_showers_per_run : int
        Number of showers in a single run.
    num_showers_target : int
        Targeted population in each allsky. It is the number of showers
        stored in the allsky.
    """
    assert num_runs >= 0
    assert num_showers_per_run >= 0
    assert num_showers_target > 0

    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    for sk in site_keys:
        sk_dir = os.path.join(work_dir, sk)
        for pk in particle_keys:
            sk_pk_dir = os.path.join(sk_dir, pk)

            print(sk, pk)

            sm = skymap.SkyMap(sk_pk_dir)
            if sm.num_showers() < num_showers_target:
                sm.populate(
                    pool=pool,
                    num_chunks=1,
                    num_jobs=num_runs,
                    num_showers_per_job=num_showers_per_run,
                )


def needs_to_run(work_dir, num_showers_target):
    num = num_showers(work_dir=work_dir)
    for sk in num:
        for pk in num[sk]:
            if num[sk][pk] < num_showers_target:
                return True
    return False


def num_showers(work_dir):
    """
    Returns the number of simulates and stored showers.

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
    """
    out = {}
    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)
    for sk in site_keys:
        out[sk] = {}
        for pk in particle_keys:
            sm = skymap.SkyMap(work_dir=os.path.join(work_dir, sk, pk))
            out[sk][pk] = sm.num_showers()
    return out


def demonstrate(
    work_dir,
    out_dir,
    pool,
    num_jobs,
    queries=None,
    cherenkov_field_of_view_half_angle_rad=np.deg2rad(3.25),
    threshold_cherenkov_density_per_sr=5e3,
    map_key="cherenkov_to_primary",
    num=64,
):
    """
    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
    out_dir : str
        Here the demonstration plots and comments will be written to.
    cherenkov_field_of_view_half_angle_rad : float
        Include showers which emit Cherenkov light within this view-cone.
    random_seed : int
        Seed for the random requests for the query.
    num : int
        Make this many examples for each combination of site and particle.
    """
    os.makedirs(out_dir, exist_ok=True)
    assert cherenkov_field_of_view_half_angle_rad > 0

    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    for sk in site_keys:
        site_path = os.path.join(out_dir, "{:s}.json".format(sk))
        sk_dir = os.path.join(work_dir, sk)
        out_sk_dir = os.path.join(out_dir, sk)
        os.makedirs(out_sk_dir, exist_ok=True)

        for pk in particle_keys:
            sk_pk_dir = os.path.join(sk_dir, pk)
            out_sk_pk_dir = os.path.join(out_sk_dir, pk)
            os.makedirs(out_sk_pk_dir, exist_ok=True)

            sm = skymap.SkyMap(work_dir=sk_pk_dir)

            sm.demonstrate(
                path=out_sk_pk_dir,
                pool=pool,
                num_jobs=num_jobs,
                queries=queries,
                map_key="cherenkov_to_primary",
                threshold_cherenkov_density_per_sr=threshold_cherenkov_density_per_sr,
                solid_angle_sr=solid_angle_sr,
                video=True,
            )


def run_plot(work_dir, pool):
    az = np.deg2rad(120)
    zd = np.deg2rad(25)
    ha = np.deg2rad(3.25)
    max_zenith_rad = np.deg2rad(45)

    plots_dir = os.path.join(work_dir, "plot")
    jobs = []
    jobs += _plot_deflection_vs_energy_on_sky_make_jobs(
        work_dir=work_dir,
        out_dir=os.path.join(plots_dir, "deflection_vs_energy_on_sky"),
        half_angle_rad=np.deg2rad(15),
    )
    jobs += _plot_primary_deflection_make_jobs(
        work_dir=work_dir,
        out_dir=os.path.join(plots_dir, "primary_deflection"),
        azimuth_rad=az,
        zenith_rad=zd,
        half_angle_rad=ha,
    )
    jobs += _plot_cherenkov_pool_statistics_make_jobs(
        work_dir=work_dir,
        out_dir=os.path.join(plots_dir, "cherenkov_pool_statistics"),
        azimuth_rad=az,
        zenith_rad=zd,
        half_angle_rad=5 * ha,
    )
    jobs += _plot_particle_containment_quantile(
        work_dir=work_dir,
        out_dir=os.path.join(plots_dir, "particle_containment_quantile"),
        max_zenith_rad=max_zenith_rad,
        half_angle_rad=ha,
    )
    pool.map(utils.scripts.run_script_job, jobs)


def _plot_deflection_vs_energy_on_sky_make_jobs(
    work_dir, out_dir, half_angle_rad
):
    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    jobs = []
    for sk in site_keys:
        for pk in particle_keys:
            job = {
                "script": os.path.join(
                    "skymap", "scripts", "plot_deflection_vs_energy_on_sky"
                ),
                "argv": [
                    "--skymap_dir",
                    os.path.join(work_dir, sk, pk),
                    "--out_dir",
                    out_dir,
                    "--half_angle_deg",
                    str(np.rad2deg(half_angle_rad)),
                ],
            }

            result_path = os.path.join(out_dir, "{:s}_{:s}.jpg".format(sk, pk))
            if not os.path.exists(result_path):
                jobs.append(job)

    return jobs


def _plot_primary_deflection_make_jobs(
    work_dir, out_dir, azimuth_rad, zenith_rad, half_angle_rad
):
    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    jobs = []
    for sk in site_keys:
        for pk in particle_keys:
            job = {
                "script": os.path.join(
                    "skymap", "scripts", "plot_primary_deflection"
                ),
                "argv": [
                    "--skymap_dir",
                    os.path.join(work_dir, sk, pk),
                    "--out_dir",
                    out_dir,
                    "--azimuth_deg",
                    str(np.rad2deg(azimuth_rad)),
                    "--zenith_deg",
                    str(np.rad2deg(zenith_rad)),
                    "--half_angle_deg",
                    str(np.rad2deg(half_angle_rad)),
                ],
            }
            result_path = os.path.join(
                out_dir, "{:s}_{:s}_wide.jpg".format(sk, pk)
            )
            if not os.path.exists(result_path):
                jobs.append(job)

    return jobs


def _plot_cherenkov_pool_statistics_make_jobs(
    work_dir, out_dir, azimuth_rad, zenith_rad, half_angle_rad
):
    site_keys, _ = find_site_and_particle_keys(work_dir=work_dir)

    jobs = []
    for sk in site_keys:
        job = {
            "script": os.path.join(
                "skymap", "scripts", "plot_cherenkov_pool_statistics"
            ),
            "argv": [
                "--site_dir",
                os.path.join(
                    work_dir,
                    sk,
                ),
                "--out_dir",
                out_dir,
                "--azimuth_deg",
                str(np.rad2deg(azimuth_rad)),
                "--zenith_deg",
                str(np.rad2deg(zenith_rad)),
                "--half_angle_deg",
                str(np.rad2deg(half_angle_rad)),
            ],
        }
        result_path = os.path.join(
            out_dir, "{:s}_cherenkov_altitude_p50_m.jpg".format(sk)
        )
        if not os.path.exists(result_path):
            jobs.append(job)

    return jobs


def _plot_particle_containment_quantile(
    work_dir,
    out_dir,
    max_zenith_rad,
    half_angle_rad,
):
    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    jobs = []
    for sk in site_keys:
        for pk in particle_keys:
            job_result_path = os.path.join(
                out_dir, f"{sk:s}_{pk:s}_containment_vs_solid_angle.jpg"
            )
            job = {
                "script": os.path.join(
                    "skymap", "scripts", "plot_particle_containment_quantile"
                ),
                "argv": [
                    "--skymap_dir",
                    os.path.join(
                        work_dir,
                        sk,
                        pk,
                    ),
                    "--out_dir",
                    out_dir,
                    "--max_zenith_deg",
                    str(np.rad2deg(max_zenith_rad)),
                    "--half_angle_deg",
                    str(np.rad2deg(half_angle_rad)),
                ],
            }
            if not os.path.exists(job_result_path):
                jobs.append(job)

    return jobs
