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


def init(
    work_dir,
    energy_stop_GeV=64.0,
    site_keys=None,
    particle_keys=None,
):
    """
    Creates tables listing the magnetic deflection of atmospheric showers
    induced by particles. The tables cover the all the sky (see
    magnetic_deflection.allsky).

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
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
    if site_keys is None:
        site_keys = atmospheric_cherenkov_response.sites.keys()
    if particle_keys is None:
        particle_keys = atmospheric_cherenkov_response.particles.keys()

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

                energy_start_GeV = (
                    atmospheric_cherenkov_response.particles.compile_energy(
                        particle["population"]["energy"]["start_GeV"]
                    )
                )

                allsky.init(
                    work_dir=sk_pk_dir,
                    particle_key=pk,
                    site_key=sk,
                    energy_start_GeV=energy_start_GeV,
                    energy_stop_GeV=energy_stop_GeV,
                )


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
            sky = allsky.AllSky(sk_pk_dir)
            if sky.num_showers() < num_showers_target:
                sky.populate(
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
            sky = allsky.AllSky(os.path.join(work_dir, sk, pk))
            out[sk][pk] = sky.num_showers()
    return out


def export_csv(work_dir, out_dir, fraction=1.0):
    """
    Exports all deflection tables in the work_dir into csv-fiels.

    Parameters
    ----------
    work_dir : str
        Contains the site-dirs which in turn contain the particle-dirs.
    out_dir : str
        Here the csv-tables will be written to.
    fraction : float
        Export only this random fraction.

    example
    -------

    |-> out_dir
            |-> namibia
                    |-> electron
                            |-> config.json
                            |-> showers.csv


    """
    os.makedirs(out_dir, exist_ok=True)

    site_keys, particle_keys = find_site_and_particle_keys(work_dir=work_dir)

    for sk in site_keys:
        sk_dir = os.path.join(work_dir, sk)
        out_sk_dir = os.path.join(out_dir, sk)
        os.makedirs(out_sk_dir, exist_ok=True)

        for pk in particle_keys:
            sk_pk_dir = os.path.join(sk_dir, pk)
            out_sk_pk_dir = os.path.join(out_sk_dir, pk)
            os.makedirs(out_sk_pk_dir, exist_ok=True)

            print(sk, pk)
            sky = allsky.AllSky(sk_pk_dir)

            config_path = os.path.join(out_sk_pk_dir, "config.json")

            with rnw.open(config_path, "wt") as f:
                f.write(json_numpy.dumps(sky.config["site"], indent=4))

            table_path = os.path.join(out_sk_pk_dir, "showers.csv")
            if not os.path.exists(table_path):
                sky.store.export_csv(path=table_path, fraction=fraction)


def demonstrate_query(
    work_dir,
    out_dir,
    cherenkov_field_of_view_half_angle_rad=np.deg2rad(6.5),
    random_seed=43,
    num=64,
):
    """
    Creates a plots which demonstrate the query process.
    For each (site,particle) combination a random query will be performed in
    order to obtain the direction of the primary particle to create Cherenkov
    light within the instrument's field-of-view. The instruments pointing and
    the energy of the primary particle are drawn randomly.

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

    hemisphere_grid = allsky.hemisphere.Grid(num_vertices=4069)
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

            allsky_deflection = allsky.AllSky(sk_pk_dir)

            particle_scatter_half_angle_rad = np.deg2rad(
                allsky_deflection.config["particle"]["population"][
                    "direction"
                ]["scatter_cone_half_angle_deg"]
            )

            for q in range(num):
                print(sk, pk, q)
                prng = np.random.Generator(np.random.PCG64(q))
                qkey = "{:06d}".format(q)

                request = _draw_shower_request(
                    prng=prng,
                    allsky_deflection=allsky_deflection,
                )
                request[
                    "cherenkov_field_of_view_half_angle_rad"
                ] = cherenkov_field_of_view_half_angle_rad

                request_path = os.path.join(
                    out_sk_pk_dir, qkey + "_request.json"
                )
                cone_path = os.path.join(out_sk_pk_dir, qkey + "_cone.json")
                grid_path = os.path.join(out_sk_pk_dir, qkey + "_grid.json")

                with rnw.open(request_path, "wt") as f:
                    f.write(json_numpy.dumps(request))

                if os.path.exists(cone_path) and os.path.exists(grid_path):
                    continue

                with rnw.open(request_path, "wt") as f:
                    f.write(json_numpy.dumps(request))

                (
                    res_cone,
                    dbg_cone,
                ) = allsky.random.draw_particle_direction_with_cone(
                    prng=prng,
                    azimuth_rad=request["cherenkov_azimuth_rad"],
                    zenith_rad=request["cherenkov_zenith_rad"],
                    half_angle_rad=cherenkov_field_of_view_half_angle_rad,
                    energy_GeV=request["primary_particle_energy_GeV"],
                    shower_spread_half_angle_rad=particle_scatter_half_angle_rad,
                    min_num_cherenkov_photons=1e2,
                    allsky_deflection=allsky_deflection,
                )
                with rnw.open(cone_path, "wt") as f:
                    f.write(
                        json_numpy.dumps(
                            {"result": res_cone, "debug": dbg_cone}
                        )
                    )

                (
                    res_grid,
                    dbg_grid,
                ) = allsky.random.draw_particle_direction_with_masked_grid(
                    prng=prng,
                    azimuth_rad=request["cherenkov_azimuth_rad"],
                    zenith_rad=request["cherenkov_zenith_rad"],
                    half_angle_rad=cherenkov_field_of_view_half_angle_rad,
                    energy_GeV=request["primary_particle_energy_GeV"],
                    shower_spread_half_angle_rad=particle_scatter_half_angle_rad,
                    min_num_cherenkov_photons=1e2,
                    allsky_deflection=allsky_deflection,
                    hemisphere_grid=hemisphere_grid,
                )
                with rnw.open(grid_path, "wt") as f:
                    f.write(
                        json_numpy.dumps(
                            {"result": res_grid, "debug": dbg_grid}
                        )
                    )

                allsky.random.plot_cone(
                    result=res_cone,
                    debug=dbg_cone,
                    path=os.path.join(
                        out_sk_pk_dir, "{:06d}_cone.svg".format(q)
                    ),
                )
                allsky.random.plot_masked_grid(
                    result=res_grid,
                    debug=dbg_grid,
                    path=os.path.join(
                        out_sk_pk_dir, "{:06d}_grid.svg".format(q)
                    ),
                    hemisphere_grid=hemisphere_grid,
                )

                try:
                    for method_key in ["cone", "grid"]:
                        ipath = os.path.join(
                            out_sk_pk_dir,
                            "{:06d}_{:s}.svg".format(q, method_key),
                        )
                        opath = os.path.join(
                            out_sk_pk_dir,
                            "{:06d}_{:s}.jpg".format(q, method_key),
                        )
                        subprocess.call(["convert", ipath, opath])
                        os.remove(ipath)
                except:
                    pass


def _draw_shower_request(prng, allsky_deflection):
    """
    Make the parameters for a random query to be performed on an AllSky.
    """
    (
        cer_az_rad,
        cer_zd_rad,
    ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
        prng=prng,
        azimuth_rad=0.0,
        zenith_rad=0.0,
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=allsky_deflection.config["binning"][
            "direction"
        ]["particle_max_zenith_distance_rad"],
    )
    energy_GeV = corsika_primary.random.distributions.draw_power_law(
        prng=prng,
        lower_limit=allsky_deflection.config["binning"]["energy"]["start_GeV"],
        upper_limit=allsky_deflection.config["binning"]["energy"]["stop_GeV"],
        power_slope=-2.0,
        num_samples=1,
    )[0]
    return {
        "cherenkov_azimuth_rad": cer_az_rad,
        "cherenkov_zenith_rad": cer_zd_rad,
        "primary_particle_energy_GeV": energy_GeV,
    }
