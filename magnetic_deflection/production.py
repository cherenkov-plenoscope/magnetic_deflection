import atmospheric_cherenkov_response as acr
from . import allsky
import os
import numpy as np
import json_numpy
import rename_after_writing as rnw
import corsika_primary
from svg_cartesian_plot import inkscape
import subprocess


def init(
    work_dir,
    energy_stop_GeV=64.0,
    site_keys=atmospheric_cherenkov_response.sites.keys(),
    particle_keys=atmospheric_cherenkov_response.particles.keys(),
    corsika_primary_path=allsky.production.default_corsika_primary_mod_path(),
):
    """
    Creates tables listing the magnetic deflection of atmospheric showers
    induced by particles. The tables cover the all the sky (see
    magnetic_deflection.allsky).

    Parameters
    ----------
    work_dir : str
        The directory to contain the sites and the particles.
        Path will be: work_dir/site/particle
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
    corsika_primary_path : str
        Path to the CORSIKA-primary-mod executable.

    example
    -------
    when site_keys is [namibia, chile] and
    particle_keys is [electron, proton, gamma], the work_dir
    will look like this:

    work_dir
         |
         |_namibia
         |      |_electron  <- each of these directories is an AllSky.
         |      |_proton
         |      |_gamma
         |
         |___chile
                |_electron
                |_proton
                |_gamma

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
                    corsika_primary_path=corsika_primary_path,
                )


def run(work_dir, pool, num_runs=960, num_showers_per_run=1280):
    assert num_runs >= 0
    assert num_showers_per_run >= 0

    for sk in acr.sites.keys():
        sk_dir = os.path.join(work_dir, sk)
        for pk in acr.particles.keys():
            sk_pk_dir = os.path.join(sk_dir, pk)

            print(sk, pk)
            sky = allsky.open(sk_pk_dir)
            sky.populate(
                pool=pool,
                num_chunks=1,
                num_jobs=num_runs,
                num_showers_per_job=num_showers_per_run,
            )


def export_csv(work_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for sk in acr.sites.keys():
        site_path = os.path.join(out_dir, "{:s}.json".format(sk))
        sk_dir = os.path.join(work_dir, sk)
        for pk in acr.particles.keys():
            sk_pk_dir = os.path.join(sk_dir, pk)

            print(sk, pk)
            sky = allsky.open(sk_pk_dir)

            with rnw.open(site_path, "wt") as f:
                f.write(json_numpy.dumps(sky.config["site"], indent=4))

            out_path = os.path.join(out_dir, "{:s}_{:s}.csv".format(sk, pk))
            if not os.path.exists(out_path):
                sky.store.export_csv(path=out_path)


def demonstrate_query(
    work_dir,
    out_dir,
    cherenkov_field_of_view_half_angle_deg=6.5,
    random_seed=43,
    num=64,
):
    os.makedirs(out_dir, exist_ok=True)
    assert cherenkov_field_of_view_half_angle_deg > 0

    hemisphere_grid = allsky.hemisphere.Grid(num_vertices=4069)

    for sk in acr.sites.keys():
        site_path = os.path.join(out_dir, "{:s}.json".format(sk))
        sk_dir = os.path.join(work_dir, sk)
        out_sk_dir = os.path.join(out_dir, sk)
        os.makedirs(out_sk_dir, exist_ok=True)

        for pk in acr.particles.keys():
            sk_pk_dir = os.path.join(sk_dir, pk)
            out_sk_pk_dir = os.path.join(out_sk_dir, pk)
            os.makedirs(out_sk_pk_dir, exist_ok=True)

            allsky_deflection = allsky.open(sk_pk_dir)

            particle_scatter_half_angle_deg = allsky_deflection.config[
                "particle"
            ]["population"]["direction"]["scatter_cone_half_angle_deg"]

            for q in range(num):
                print(sk, pk, q)
                prng = np.random.Generator(np.random.PCG64(q))
                qkey = "{:06d}".format(q)

                request = _draw_shower_request(
                    prng=prng,
                    allsky_deflection=allsky_deflection,
                )
                request[
                    "cherenkov_field_of_view_half_angle_deg"
                ] = cherenkov_field_of_view_half_angle_deg

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
                    azimuth_deg=request["cherenkov_azimuth_deg"],
                    zenith_deg=request["cherenkov_zenith_deg"],
                    half_angle_deg=cherenkov_field_of_view_half_angle_deg,
                    energy_GeV=request["primary_particle_energy_GeV"],
                    shower_spread_half_angle_deg=particle_scatter_half_angle_deg,
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
                    azimuth_deg=request["cherenkov_azimuth_deg"],
                    zenith_deg=request["cherenkov_zenith_deg"],
                    half_angle_deg=cherenkov_field_of_view_half_angle_deg,
                    energy_GeV=request["primary_particle_energy_GeV"],
                    shower_spread_half_angle_deg=particle_scatter_half_angle_deg,
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
    (
        cer_az_rad,
        cer_zd_rad,
    ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
        prng=prng,
        azimuth_rad=0.0,
        zenith_rad=0.0,
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=np.deg2rad(
            allsky_deflection.config["binning"]["direction"][
                "particle_max_zenith_distance_deg"
            ]
        ),
    )
    energy_GeV = corsika_primary.random.distributions.draw_power_law(
        prng=prng,
        lower_limit=allsky_deflection.config["binning"]["energy"]["start_GeV"],
        upper_limit=allsky_deflection.config["binning"]["energy"]["stop_GeV"],
        power_slope=-2.0,
        num_samples=1,
    )[0]
    return {
        "cherenkov_azimuth_deg": np.rad2deg(cer_az_rad),
        "cherenkov_zenith_deg": np.rad2deg(cer_zd_rad),
        "primary_particle_energy_GeV": energy_GeV,
    }
