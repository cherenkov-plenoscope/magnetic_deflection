from . import discovery
from . import map_and_reduce
from . import examples
from . import analysis
from . import light_field_characterization
from . import corsika
from . import spherical_coordinates
from . import tools
from . import recarray_io
from . import Records
from . import debug
from . import work_dir_structure

import os
import pandas
import numpy as np
import pkg_resources
import subprocess
import glob
import atmospheric_cherenkov_response
import json_line_logger as jlogging


def init(
    work_dir,
    particles=None,
    sites=None,
    pointing=None,
    max_energy=64.0,
    num_energy_supports=16,
):
    """
    Make work_dir
    """

    if particles == None:
        particles = {}
        for pk in atmospheric_cherenkov_response.particles.keys():
            particles[pk] = atmospheric_cherenkov_response.particles.init(pk)

    if sites == None:
        sites = {}
        for sk in atmospheric_cherenkov_response.sites.keys():
            sites[pk] = atmospheric_cherenkov_response.sites.init(sk)

    if pointing == None:
        pointing = atmospheric_cherenkov_response.pointing.init(
            azimuth_deg=0.0, zenith_deg=0.0
        )

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)

    tools.write_json(os.path.join(work_dir, "config", "sites.json"), sites)
    tools.write_json(
        os.path.join(work_dir, "config", "pointing.json"), pointing
    )
    tools.write_json(
        os.path.join(work_dir, "config", "particles.json"), particles
    )
    _write_default_config(
        path=os.path.join(work_dir, "config", "config.json"),
        energy_supports_max=max_energy,
        energy_supports_num=num_energy_supports,
    )
    _write_default_plotting_config(
        path=os.path.join(work_dir, "config", "plotting.json"),
    )


def read_config(work_dir, keys=work_dir_structure.all_config_keys()):
    return work_dir_structure.read_config(work_dir=work_dir, keys=keys)


def _write_default_config(path, energy_supports_max, energy_supports_num):
    cfg = {
        "energy_supports_max": float(energy_supports_max),
        "energy_supports_num": int(energy_supports_num),
        "energy_supports_power_law_slope": -1.7,
        "discovery_max_total_energy": 8e3,
        "discovery_min_energy_per_iteration": 16.0,
        "discovery_min_num_showers_per_iteration": 32,
        "statistics_total_energy": 8e3,
        "statistics_min_num_showers": 500,
        "min_num_cherenkov_photons": 100,
        "corsika_primary_path": examples.CORSIKA_PRIMARY_MOD_PATH,
        "statistics_r_bin_edges": np.linspace(0.0, 24 * 25.0, 25),
        "statistics_theta_bin_edges_deg": np.linspace(0.0, 6.0, 13),
    }
    tools.write_json(path, obj=cfg)


def _write_default_plotting_config(path):
    tools.write_json(
        path=path, obj=examples.PLOTTING,
    )


def make_jobs(work_dir, logger=None):
    logger = logger if logger else jlogging.LoggerStdout()
    CFG = read_config(work_dir, ["sites", "particles", "pointing", "config",])

    jobs = []
    job_id = 1  # start at 1
    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            logger.info("make_jobs: {:s} {:s}".format(skey, pkey))
            site_particle_jobs = map_and_reduce.make_jobs(
                first_job_id=job_id,
                work_dir=work_dir,
                site=CFG["sites"][skey],
                site_key=skey,
                particle=CFG["particles"][pkey],
                particle_key=pkey,
                pointing=CFG["pointing"],
                energy_supports_min=atmospheric_cherenkov_response.particles.compile_energy(
                    CFG["particles"][pkey]["population"]["energy"]["start_GeV"]
                ),
                **CFG["config"],
            )
            job_id += len(site_particle_jobs)
            jobs += site_particle_jobs

    logger.info("make_jobs: sort by energy ascending")
    return tools.sort_records_by_key(
        records=jobs, keys=("particle", "energy_GeV")
    )


def reduce(work_dir, logger=jlogging.LoggerStdout()):
    logger.info("reduce")
    reduce_raw_deflection(work_dir=work_dir, logger=logger)
    analyse_raw_deflection(
        work_dir=work_dir, min_fit_energy=0.65, logger=logger,
    )
    reduce_steerings(work_dir=work_dir, logger=logger)
    reduce_statistics(work_dir=work_dir, logger=logger)


def reduce_raw_deflection(work_dir, logger=jlogging.LoggerStdout()):
    CFG = read_config(work_dir, ["sites", "particles",])
    map_basenames_wildcard = work_dir_structure.map_basenames_wildcard()

    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            os.makedirs(
                os.path.join(work_dir, "reduce", skey, pkey, ".deflection"),
                exist_ok=True,
            )

            out_path = os.path.join(
                work_dir, "reduce", skey, pkey, ".deflection", "raw.csv"
            )
            if not os.path.exists(out_path):
                logger.info(
                    "reduce: raw_deflection {:s} {:s} make new".format(
                        skey, pkey
                    )
                )
                paths = glob.glob(
                    os.path.join(
                        work_dir,
                        "map",
                        skey,
                        pkey,
                        map_basenames_wildcard["deflection"],
                    )
                )
                raw = _reduce_raw_deflection_site_particle(paths=paths)
                recarray_io.write_to_csv(
                    recarray=raw, path=out_path,
                )
            else:
                logger.info(
                    "reduce: raw_deflection {:s} {:s} already done".format(
                        skey, pkey
                    )
                )


def _reduce_raw_deflection_site_particle(paths):
    raw = []
    for path in paths:
        raw.append(tools.read_json(path))
    raw_df = pandas.DataFrame(raw)
    raw_rec = raw_df.to_records(index=False)
    del raw_df
    order = np.argsort(raw_rec["particle_energy_GeV"])
    raw_rec = raw_rec[order]
    return raw_rec


def reduce_steerings(work_dir, logger=jlogging.LoggerStdout()):
    CFG = read_config(work_dir, ["sites", "particles",])
    map_basenames_wildcard = work_dir_structure.map_basenames_wildcard()
    reduce_basenames = work_dir_structure.reduce_basenames()

    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            os.makedirs(
                os.path.join(work_dir, "reduce", skey, pkey), exist_ok=True
            )

            out_steering_path = os.path.join(
                work_dir,
                "reduce",
                skey,
                pkey,
                reduce_basenames["statistics_steering"],
            )
            if not os.path.exists(out_steering_path):
                logger.info(
                    "reduce: steerings {:s} {:s} make new".format(skey, pkey)
                )
                paths = glob.glob(
                    os.path.join(
                        work_dir,
                        "map",
                        skey,
                        pkey,
                        map_basenames_wildcard["statistics_steering"],
                    )
                )
                steerings_and_seeds = _reduce_statistics_steering_site_particle(
                    paths=paths, skey=skey, pkey=pkey, logger=logger,
                )
                expl = corsika.cpw.steering.write_steerings_and_seeds(
                    runs=steerings_and_seeds, path=out_steering_path,
                )
            else:
                logger.info(
                    "reduce: steerings {:s} {:s} already done".format(
                        skey, pkey
                    )
                )


def reduce_statistics(work_dir, logger=jlogging.LoggerStdout()):
    CFG = read_config(work_dir, ["sites", "particles",])
    map_basenames_wildcard = work_dir_structure.map_basenames_wildcard()
    reduce_basenames = work_dir_structure.reduce_basenames()

    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            os.makedirs(
                os.path.join(work_dir, "reduce", skey, pkey), exist_ok=True
            )

            out_statistics_path = os.path.join(
                work_dir, "reduce", skey, pkey, reduce_basenames["statistics"],
            )
            if not os.path.exists(out_statistics_path):
                logger.info(
                    "reduce: statistics {:s} {:s} make new".format(skey, pkey)
                )
                paths = glob.glob(
                    os.path.join(
                        work_dir,
                        "map",
                        skey,
                        pkey,
                        map_basenames_wildcard["statistics"],
                    )
                )
                shower_statistics = _reduce_statistics_site_particle(
                    paths=paths, skey=skey, pkey=pkey, logger=logger,
                )
                recarray_io.write_to_tar(
                    recarray=shower_statistics, path=out_statistics_path,
                )
            else:
                logger.info(
                    "reduce: statistics {:s} {:s} already done".format(
                        skey, pkey
                    )
                )


def read_statistics(work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])
    reduce_basenames = work_dir_structure.reduce_basenames()
    out = {}
    for skey in CFG["sites"]:
        out[skey] = {}
        for pkey in CFG["particles"]:
            path = os.path.join(
                work_dir, "reduce", skey, pkey, reduce_basenames["statistics"],
            )
            out[skey][pkey] = recarray_io.read_from_tar(path=path)
    return out


def read_explicit_steerings(work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])
    reduce_basenames = work_dir_structure.reduce_basenames()
    out = {}
    for skey in CFG["sites"]:
        out[skey] = {}
        for pkey in CFG["particles"]:
            path = os.path.join(
                work_dir,
                "reduce",
                skey,
                pkey,
                reduce_basenames["statistics_steering"],
            )
            out[skey][pkey] = corsika.cpw.steering.read_steerings_and_seeds(
                path
            )
    return out


def analyse_raw_deflection(
    work_dir, min_fit_energy=0.65, logger=jlogging.LoggerStdout(),
):
    CFG = read_config(work_dir, ["sites", "particles",])
    PARTICLES = CFG["particles"]
    SITES = CFG["sites"]

    min_particle_energy = np.min(
        [np.min(PARTICLES[p]["energy_bin_edges_GeV"]) for p in PARTICLES]
    )
    if min_particle_energy > min_fit_energy:
        min_fit_energy = 1.1 * min_particle_energy

    for skey in SITES:
        for pkey in PARTICLES:
            wrsp = os.path.join(work_dir, "reduce", skey, pkey, ".deflection")

            stages = {
                "raw": os.path.join(wrsp, "raw.csv"),
                "raw_valid": os.path.join(wrsp, "raw_valid.csv"),
                "raw_valid_add": os.path.join(wrsp, "raw_valid_add.csv"),
                "raw_valid_add_clean": os.path.join(
                    wrsp, "raw_valid_add_clean.csv"
                ),
                "raw_valid_add_clean_high": os.path.join(
                    wrsp, "raw_valid_add_clean_high.csv"
                ),
                "raw_valid_add_clean_high_power": os.path.join(
                    wrsp, "raw_valid_add_clean_high_power.json"
                ),
                "result": os.path.join(
                    work_dir, "reduce", skey, pkey, "deflection.csv"
                ),
            }

            if os.path.exists(stages["result"]):
                logger.info(
                    "reduce: analyse {:s} {:s} already done".format(skey, pkey)
                )
                continue
            else:
                logger.info(
                    "reduce: analyse {:s} {:s} make new".format(skey, pkey)
                )

            charge_sign = np.sign(PARTICLES[pkey]["electric_charge_qe"])

            raw = recarray_io.read_from_csv(path=stages["raw"])

            # cut invalid
            # -----------
            raw_valid = analysis.deflection_cut_invalid(
                deflection=raw, min_energy=min_fit_energy,
            )
            recarray_io.write_to_csv(
                raw_valid, path=stages["raw_valid"],
            )

            # add density fields
            # ------------------
            raw_valid_add = analysis.deflection_add_density_fields(
                deflection=raw_valid
            )
            recarray_io.write_to_csv(
                raw_valid_add, path=stages["raw_valid_add"],
            )

            # smooth_and_reject_outliers
            # --------------------------
            raw_valid_add_clean = analysis.deflection_smooth_when_possible(
                deflection=raw_valid_add, logger=logger,
            )
            recarray_io.write_to_csv(
                raw_valid_add_clean, path=stages["raw_valid_add_clean"],
            )

            # add_high_energies
            # -----------------
            raw_valid_add_clean_high = analysis.deflection_extend_to_high_energy(
                deflection=raw_valid_add_clean,
                charge_sign=charge_sign,
                energy_start=200,
                energy_stop=600,
                num_points=20,
            )
            recarray_io.write_to_csv(
                raw_valid_add_clean_high,
                path=stages["raw_valid_add_clean_high"],
            )

            # fit_power_law
            # -------------
            power_law_fit = analysis.deflection_fit_power_law(
                deflection=raw_valid_add_clean_high,
                charge_sign=charge_sign,
                logger=logger,
            )
            tools.write_json(
                path=stages["raw_valid_add_clean_high_power"],
                obj=power_law_fit,
            )

            # export table
            # ------------
            interpolated_deflection = analysis.power_law_fit_evaluate(
                power_law_fit=power_law_fit,
                particle=CFG["particles"][pkey],
                num_supports=1024,
            )
            recarray_io.write_to_csv(
                recarray=interpolated_deflection, path=stages["result"],
            )

    script_path = os.path.abspath(
        pkg_resources.resource_filename(
            "magnetic_deflection",
            os.path.join("scripts", "plot_deflection.py"),
        )
    )
    subprocess.call(["python", script_path, work_dir])


def read_deflection(work_dir, style="dict"):
    CFG = read_config(work_dir, ["sites", "particles",])
    mag = {}
    for skey in CFG["sites"]:
        mag[skey] = {}
        for pkey in CFG["particles"]:
            path = os.path.join(
                work_dir, "reduce", skey, pkey, "deflection.csv",
            )
            df = pandas.read_csv(path)
            if style == "record":
                mag[skey][pkey] = df.to_records()
            elif style == "dict":
                mag[skey][pkey] = df.to_dict("list")
            else:
                raise KeyError("Unknown style: '{:s}'.".format(style))
    return mag


def _reduce_statistics_site_particle(
    paths, skey=None, pkey=None, logger=jlogging.LoggerStdout()
):
    probe_recarray = recarray_io.read_from_tar(path=paths[0])
    probe_dtypes = Records.get_dtypes_from_numpy_recarray(
        recarray=probe_recarray
    )
    stats = Records.init(dtypes=probe_dtypes)
    del probe_recarray
    del probe_dtypes

    num = len(paths)
    for i, path in enumerate(paths):
        basename = os.path.basename(path)
        job_id = int(basename[0:6])
        logger.info(
            "reduce: statistics {:s} {:s} ".format(skey, pkey)
            + "run {:06d}, {: 6d} / {: 6d}".format(job_id, i, num)
        )
        pools = recarray_io.read_from_tar(path=path)
        stats = Records.append_numpy_recarray(stats, pools)

    return Records.to_numpy_recarray(stats)


def _reduce_statistics_steering_site_particle(
    paths, skey=None, pkey=None, logger=jlogging.LoggerStdout()
):
    bundle = {}
    num = len(paths)
    for i, path in enumerate(paths):
        basename = os.path.basename(path)
        run_id = int(basename[0:6])
        logger.info(
            "reduce: steerings {:s} {:s} ".format(skey, pkey)
            + "run {:06d}, {: 6d} / {: 6d}".format(run_id, i, num),
        )
        runs = corsika.cpw.steering.read_steerings_and_seeds(path=path)
        bundle[run_id] = runs[run_id]
    return bundle
