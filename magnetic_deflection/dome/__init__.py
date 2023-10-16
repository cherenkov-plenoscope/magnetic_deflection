import numpy as np
import os
import atmospheric_cherenkov_response
import json_utils
import rename_after_writing as rnw
import corsika_primary as cpw
import pandas
import glob
from . import binning
from .. import corsika
from .. import spherical_coordinates
from .. import examples

"""
Goals
-----
-   Answer from what direction (A) a particle needs to be thrown in order to
    see its Cherenkov-light from direction (B).
"""


def init(
    work_dir,
    site=None,
    particle=None,
    direction_max_zenith_distance_deg=60,
    direction_num_bins=256,
    energy_start_GeV=0.25,
    energy_stop_GeV=64,
    energy_num_bins=32,
    corsika_primary_path=None,
):
    os.makedirs(work_dir, exist_ok=True)
    join = os.path.join

    if site == None:
        site = atmospheric_cherenkov_response.sites.init("lapalma")

    if particle == None:
        particle = atmospheric_cherenkov_response.particles.init("electron")

    cfg_dir = join(work_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)

    with rnw.open(join(cfg_dir, "site.json"), "wt") as f:
        f.write(json_utils.dumps(site, indent=4))

    with rnw.open(join(cfg_dir, "particle.json"), "wt") as f:
        f.write(json_utils.dumps(particle, indent=4))

    if corsika_primary_path == None:
        corsika_primary_path = examples.CORSIKA_PRIMARY_MOD_PATH

    with rnw.open(join(cfg_dir, "executables.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                {"corsika_primary_path": corsika_primary_path}, indent=4
            )
        )

    dome_dir = join(work_dir, "dome")
    binning.init_dome(
        dome_dir=dome_dir,
        direction_max_zenith_distance_deg=direction_max_zenith_distance_deg,
        direction_num_bins=direction_num_bins,
        energy_start_GeV=energy_start_GeV,
        energy_stop_GeV=energy_stop_GeV,
        energy_num_bins=energy_num_bins,
    )


def status(work_dir):
    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    dome = binning.Dome(os.path.join(work_dir, "dome"))

    s = "Dome\n"
    s += "====\n"
    s += "site: {:s}\n".format(config["site"]["key"])
    s += "particle: {:s}\n".format(config["particle"]["key"])

    print(s)


def commit_stage(work_dir):
    jobs = _commit_stage_make_jobs(work_dir=work_dir)
    for job in jobs:
        _commit_stage_run_job(job=job)


def _commit_stage_make_jobs(work_dir):
    jobs = []
    stage = _find_files_in_stage(dome_dir=os.path.join(work_dir, "dome"))
    for domebin in stage:
        for key in ["cherenkov", "primaries"]:
            if len(stage[domebin][key]) > 0:
                job = {
                    "work_dir": work_dir,
                    "domebin": domebin,
                    "paths": stage[domebin][key],
                    "key": key,
                }
                jobs.append(job)
    return jobs


def _commit_stage_run_job(job):
    domebin_dir = binning.make_domebin_dir(job["domebin"])
    dst_path = os.path.join(
        job["work_dir"], "dome", domebin_dir, job["key"] + ".jsonl"
    )
    if not os.path.isfile(dst_path):
        with open(dst_path, "wt") as f:
            pass

    dst_tmp_path = dst_path + ".part"
    rnw.move(dst_path, dst_tmp_path)

    with open(dst_tmp_path, "at") as dst_f:
        for src_path in job["paths"]:
            with open(src_path, "rt") as src_f:
                for line in src_f.readlines():
                    dst_f.write(line)
            os.remove(src_path)
    rnw.move(dst_tmp_path, dst_path)


def _find_files_in_stage(dome_dir):
    dome = binning.Dome(dome_dir)
    stage = {}
    for domebin in dome.list_domebins():
        domebin_dir = binning.make_domebin_dir(domebin=domebin)
        cer_dir = os.path.join(
            dome_dir, domebin_dir, "stage", "cherenkov", "*.jsonl"
        )
        prm_dir = os.path.join(
            dome_dir, domebin_dir, "stage", "primaries", "*.jsonl"
        )
        stage[domebin] = {
            "cherenkov": glob.glob(cer_dir),
            "primaries": glob.glob(prm_dir),
        }
    return stage


def add_cherenkov_pools_to_stage(
    work_dir,
    run_id,
    num_showers,
):
    join = os.path.join

    cherenkov_pools, steering_dict = make_cherenkov_pools_statistics(
        work_dir=work_dir,
        run_id=run_id,
        num_showers=num_showers,
    )
    dome = binning.Dome(join(work_dir, "dome"))

    cherenkov_pools_stages = make_cherenkov_pools_stages(
        dome=dome, cherenkov_pools=cherenkov_pools
    )

    dome_dir = join(work_dir, "dome")

    for domebin in cherenkov_pools_stages:
        domebin_dir = binning.make_domebin_dir(domebin=domebin)
        out_dir = join(dome_dir, domebin_dir, "stage", "cherenkov")
        os.makedirs(out_dir, exist_ok=True)
        stage_path = join(out_dir, "{:06d}.jsonl".format(run_id))
        json_utils.lines.write(
            path=stage_path, obj_list=cherenkov_pools_stages[domebin]
        )

    primarie_stages = make_primarie_stages(
        dome=dome, steering_dict=steering_dict
    )
    for domebin in primarie_stages:
        domebin_dir = binning.make_domebin_dir(domebin=domebin)
        out_dir = join(dome_dir, domebin_dir, "stage", "primaries")
        os.makedirs(out_dir, exist_ok=True)
        stage_path = join(out_dir, "{:06d}.jsonl".format(run_id))
        json_utils.lines.write(
            path=stage_path, obj_list=primarie_stages[domebin]
        )


def make_cherenkov_pools_statistics(
    work_dir,
    run_id,
    num_showers,
):
    config = json_utils.tree.read(os.path.join(work_dir, "config"))
    dome_binning = json_utils.tree.read(
        os.path.join(work_dir, "dome", "binning")
    )

    return _make_cherenkov_pools_statistics(
        site=config["site"],
        particle_id=config["particle"]["corsika_particle_id"],
        particle_energy_start_GeV=dome_binning["energy_bin"]["start"],
        particle_energy_stop_GeV=dome_binning["energy_bin"]["stop"],
        particle_energy_power_slope=-2.0,
        particle_cone_azimuth_deg=0.0,
        particle_cone_zenith_deg=0.0,
        particle_cone_opening_angle_deg=dome_binning["direction_bin"][
            "max_zenith_distance_deg"
        ],
        num_showers=num_showers,
        min_num_cherenkov_photons=100,
        corsika_primary_path=config["executables"]["corsika_primary_path"],
        run_id=run_id,
    )


def _make_cherenkov_pools_statistics(
    site,
    particle_id,
    particle_energy_start_GeV,
    particle_energy_stop_GeV,
    particle_energy_power_slope,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
    min_num_cherenkov_photons,
    corsika_primary_path,
    run_id,
):
    steering_dict = make_corsika_primary_steering(
        run_id=run_id,
        site=site,
        particle_id=particle_id,
        particle_energy_start_GeV=particle_energy_start_GeV,
        particle_energy_stop_GeV=particle_energy_stop_GeV,
        particle_energy_power_slope=particle_energy_power_slope,
        particle_cone_azimuth_deg=particle_cone_azimuth_deg,
        particle_cone_zenith_deg=particle_cone_zenith_deg,
        particle_cone_opening_angle_deg=particle_cone_opening_angle_deg,
        num_showers=num_showers,
    )
    cherenkov_pool, _event_seeds = corsika.estimate_cherenkov_pool(
        corsika_steering_dict=steering_dict,
        corsika_primary_path=corsika_primary_path,
        min_num_cherenkov_photons=min_num_cherenkov_photons,
        statistics_optional={},
    )

    return cherenkov_pool, steering_dict


def make_primarie_stages(dome, steering_dict):
    primaries = pandas.DataFrame(steering_dict["primaries"]).to_records(
        index=False
    )
    primaries_stage = {}
    for p in range(primaries.shape[0]):
        particle_azimuth_deg = np.rad2deg(primaries["azimuth_rad"][p])
        particle_zenith_deg = np.rad2deg(primaries["zenith_rad"][p])
        domebin_dist, domebin = dome.query(
            azimuth_deg=particle_azimuth_deg,
            zenith_deg=particle_zenith_deg,
            energy_GeV=primaries["energy_GeV"][p],
        )
        if domebin not in primaries_stage:
            primaries_stage[domebin] = []
        particle_cx, particle_cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=particle_azimuth_deg, zenith_deg=particle_zenith_deg
        )
        entry = {
            "run": steering_dict["run"]["run_id"],
            "event": p,
            "particle_cx_rad": particle_cx,
            "particle_cy_rad": particle_cy,
            "particle_energy_GeV": primaries["energy_GeV"][p],
        }
        primaries_stage[domebin].append(entry)

    return primaries_stage


def make_cherenkov_pools_stages(dome, cherenkov_pools):
    pools = pandas.DataFrame(cherenkov_pools).to_records(index=False)
    cherenkov_pools_stage = {}
    for p in range(pools.shape[0]):
        cherenkov_cx_rad = pools["cherenkov_cx_rad"][p]
        cherenkov_cy_rad = pools["cherenkov_cy_rad"][p]
        (
            cherenkov_az_deg,
            cherenkov_zd_deg,
        ) = spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=cherenkov_cx_rad,
            cy=cherenkov_cy_rad,
        )
        domebin_dist, domebin = dome.query(
            azimuth_deg=cherenkov_az_deg,
            zenith_deg=cherenkov_zd_deg,
            energy_GeV=pools["particle_energy_GeV"][p],
        )
        if domebin not in cherenkov_pools_stage:
            cherenkov_pools_stage[domebin] = []

        particle_cx, particle_cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=pools["particle_azimuth_deg"][p],
            zenith_deg=pools["particle_zenith_deg"][p],
        )
        entry = {
            "run": pools["run"][p],
            "event": pools["event"][p],
            "particle_energy_GeV": pools["particle_energy_GeV"][p],
            "particle_cx_rad": particle_cx,
            "particle_cy_rad": particle_cy,
            "cherenkov_num_photons": pools["cherenkov_num_photons"][p],
            "cherenkov_num_bunches": pools["cherenkov_num_bunches"][p],
            "cherenkov_cx_rad": cherenkov_cx_rad,
            "cherenkov_cy_rad": cherenkov_cy_rad,
            "cherenkov_angle50_rad": pools["cherenkov_angle50_rad"][p],
            "cherenkov_x_m": pools["cherenkov_x_m"][p],
            "cherenkov_y_m": pools["cherenkov_y_m"][p],
            "cherenkov_radius50_m": pools["cherenkov_radius50_m"][p],
            "cherenkov_t_s": pools["cherenkov_t_s"][p],
            "cherenkov_t_std_s": pools["cherenkov_t_std_s"][p],
        }
        cherenkov_pools_stage[domebin].append(entry)

    return cherenkov_pools_stage


def make_corsika_primary_steering(
    run_id,
    site,
    particle_id,
    particle_energy_start_GeV,
    particle_energy_stop_GeV,
    particle_energy_power_slope,
    particle_cone_azimuth_deg,
    particle_cone_zenith_deg,
    particle_cone_opening_angle_deg,
    num_showers,
):
    assert run_id > 0
    i8 = np.int64
    f8 = np.float64

    prng = np.random.Generator(np.random.PCG64(run_id))

    steering = {}
    steering["run"] = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle_energy_start_GeV * 0.99),
            "stop_GeV": f8(particle_energy_stop_GeV * 1.01),
        },
        "random_seed": cpw.random.seed.make_simple_seed(seed=run_id),
    }
    steering["primaries"] = []
    for airshower_id in np.arange(1, num_showers + 1):
        az, zd = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=np.deg2rad(particle_cone_azimuth_deg),
            zenith_rad=np.deg2rad(particle_cone_zenith_deg),
            min_scatter_opening_angle_rad=np.deg2rad(0.0),
            max_scatter_opening_angle_rad=np.deg2rad(
                particle_cone_opening_angle_deg
            ),
            max_iterations=1000,
        )
        particle_energy = cpw.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=particle_energy_start_GeV,
            upper_limit=particle_energy_stop_GeV,
            power_slope=particle_energy_power_slope,
            num_samples=1,
        )
        prm = {
            "particle_id": f8(particle_id),
            "energy_GeV": f8(particle_energy),
            "zenith_rad": f8(zd),
            "azimuth_rad": f8(az),
            "depth_g_per_cm2": f8(0.0),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering
