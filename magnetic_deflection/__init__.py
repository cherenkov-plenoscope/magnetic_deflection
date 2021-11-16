from . import discovery
from . import map_and_reduce
from . import examples
from . import analysis
from . import light_field_characterization
from . import corsika
from . import spherical_coordinates
from . import tools
from . import jsonl_logger
from . import recarray_io
from . import Records
from . import debug
from . import work_dir as wdir

import os
import pandas
import numpy as np
import pkg_resources
import subprocess
import glob


def A_init_work_dir(
    particles,
    sites,
    plenoscope_pointing,
    max_energy,
    num_energy_supports,
    work_dir,
):
    """
    Make work_dir
    """
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(wdir.join(work_dir, "config"), exist_ok=True)

    tools.write_json(wdir.join(work_dir, "config", "sites.json"), sites)
    tools.write_json(wdir.join(work_dir, "config", "pointing.json"), plenoscope_pointing)
    tools.write_json(wdir.join(work_dir, "config", "particles.json"), particles)
    _write_default_config(
        path=wdir.join(work_dir, "config", "config.json"),
        energy_supports_max=max_energy,
        energy_supports_num=num_energy_supports,
    )
    _write_default_plotting_config(
        path=wdir.join(work_dir, "config", "plotting.json"),
    )


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
        "outlier_percentile": 50.0,
        "min_num_cherenkov_photons": 100,
        "corsika_primary_path": examples.CORSIKA_PRIMARY_MOD_PATH,
    }
    tools.write_json(path, obj=cfg)


def _write_default_plotting_config(path):
    tools.write_json(path=path, obj=examples.PLOTTING,)


def B_make_jobs_from_work_dir(work_dir):
    CFG = read_config(work_dir, ["sites", "particles", "pointing", "config",])

    jobs = []
    job_id = 0
    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            print("Make jobs ", skey, pkey)
            site_particle_jobs = map_and_reduce.make_jobs(
                first_job_id=job_id,
                work_dir=work_dir,
                site=CFG["sites"][skey],
                site_key=skey,
                particle=CFG["particles"][pkey],
                particle_key=pkey,
                pointing=pointing,
                energy_supports_min=min(
                    CFG["particles"][pkey]["energy_bin_edges_GeV"]
                ),
                **CFG["config"],
            )
            job_id += len(site_particle_jobs)
            jobs += site_particle_jobs

    print("Sort jobs by energy")
    return tools.sort_records_by_key(
        records=jobs, keys=("particle", "energy_GeV")
    )


def B2_read_job_results_from_work_dir(work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])

    job_results = {}
    for skey in CFG["sites"]:
        job_results[skey] = {}
        for pkey in  CFG["particles"]:
            paths = glob.glob(
                wdir.join(work_dir, "map", skey, pkey, "*_result.json")
            )
            job_results[skey][pkey] = [tools.read_json(p) for p in paths]
    return job_results


def C_reduce_job_results_in_work_dir(job_results, work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])

    for skey in CFG["sites"]:
        os.makedirs(wdir.join(work_dir, "reduce", skey), exist_ok=True)
        for pkey in CFG["particles"]:
            print("Reducing shower results: ", skey, pkey)

            df = pandas.DataFrame(job_results[skey][pkey])
            rec = df.to_records(index=False)
            order = np.argsort(rec["particle_energy_GeV"])
            rec = rec[order]
            recarray_io.write_to_csv(
                recarray=rec,
                path=wdir.join(work_dir, "reduce", skey, pkey + "_deflection.csv"),
            )


def C2_reduce_shower_statistics_in_work_dir(work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])

    for skey in CFG["sites"]:
        os.makedirs(wdir.join(work_dir, "reduce", skey), exist_ok=True)
        for pkey in CFG["particles"]:
            print("Reducing shower statistics: ", skey, pkey)

            shower_statistics = tools.read_statistics_site_particle(
                map_site_particle_dir=wdir.join(work_dir, "map", skey, pkey)
            )
            recarray_io.write_to_tar(
                recarray=shower_statistics,
                path=wdir.join(work_dir, "reduce", skey, pkey + "_shower_statistics.recarray.tar"),
            )


def read_shower_statistics(work_dir):
    CFG = read_config(work_dir, ["sites", "particles",])
    out = {}
    for skey in CFG["sites"]:
        out[skey] = {}
        for pkey in CFG["particles"]:
            out[skey][pkey] = recarray_io.read_from_tar(
                path=wdir.join(work_dir, "reduce", skey, pkey + "_shower_statistics.recarray.tar")
            )
    return out


def D_summarize_raw_deflection(
    work_dir, min_fit_energy=0.65,
):
    CFG = read_config(work_dir, ["sites", "particles",])
    PARTICLES = CFG["particles"]
    SITES = CFG["sites"]

    min_particle_energy = np.min(
        [np.min(PARTICLES[p]["energy_bin_edges_GeV"]) for p in PARTICLES]
    )
    if min_particle_energy > min_fit_energy:
        min_fit_energy = 1.1 * min_particle_energy

    raw_dir = wdir.join(work_dir, "raw")

    raw_valid_dir = wdir.join(work_dir, "raw_valid")
    os.makedirs(raw_valid_dir, exist_ok=True)

    raw_valid_add_dir = wdir.join(work_dir, "raw_valid_add")
    os.makedirs(raw_valid_add_dir, exist_ok=True)

    raw_valid_add_clean_dir = wdir.join(work_dir, "raw_valid_add_clean")
    os.makedirs(raw_valid_add_clean_dir, exist_ok=True)

    raw_valid_add_clean_high_dir = os.path.join(
        work_dir, "raw_valid_add_clean_high"
    )
    os.makedirs(raw_valid_add_clean_high_dir, exist_ok=True)

    raw_valid_add_clean_high_power_dir = os.path.join(
        work_dir, "raw_valid_add_clean_high_power"
    )
    os.makedirs(raw_valid_add_clean_high_power_dir, exist_ok=True)

    result_dir = wdir.join(work_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    for skey in SITES:
        for pkey in PARTICLES:

            fname = skey + "_" + pkey + ".csv"
            charge_sign = np.sign(PARTICLES[pkey]["electric_charge_qe"])

            print(skey, pkey, fname)

            raw = recarray_io.read_from_csv(path=os.path.join(raw_dir, fname))

            # cut invalid
            # -----------
            raw_valid = analysis.cut_invalid_from_deflection(
                deflection=raw, min_energy=min_fit_energy,
            )
            recarray_io.write_to_csv(
                raw_valid, path=os.path.join(raw_valid_dir, fname)
            )

            # add density fields
            # ------------------
            raw_valid_add = analysis.add_density_fields_to_deflection(
                deflection=raw_valid
            )
            recarray_io.write_to_csv(
                raw_valid_add, path=os.path.join(raw_valid_add_dir, fname)
            )

            # smooth_and_reject_outliers
            # --------------------------
            raw_valid_add_clean = analysis.smooth_deflection_and_reject_outliers(
                deflection=raw_valid_add
            )
            recarray_io.write_to_csv(
                raw_valid_add_clean,
                path=os.path.join(raw_valid_add_clean_dir, fname),
            )

            # add_high_energies
            # -----------------
            raw_valid_add_clean_high = analysis.add_high_energy_to_deflection(
                deflection=raw_valid_add_clean,
                charge_sign=charge_sign,
                energy_start=200,
                energy_stop=600,
                num_points=20,
            )
            recarray_io.write_to_csv(
                raw_valid_add_clean_high,
                path=os.path.join(raw_valid_add_clean_high_dir, fname),
            )

            # fit_power_law
            # -------------
            power_law_fit = analysis.fit_power_law_to_deflection(
                deflection=raw_valid_add_clean_high, charge_sign=charge_sign,
            )
            tools.write_json(
                path=os.path.join(
                    raw_valid_add_clean_high_power_dir,
                    "{:s}_{:s}.json".format(skey, pkey),
                ),
                obj=power_law_fit,
            )

            # export table
            # ------------
            interpolated_deflection = analysis.make_fit_deflection(
                power_law_fit=power_law_fit,
                particle=particles[pkey],
                num_supports=1024,
            )
            recarray_io.write_to_csv(
                recarray=interpolated_deflection,
                path=os.path.join(result_dir, fname),
            )

    script_path = os.path.abspath(
        pkg_resources.resource_filename(
            "magnetic_deflection",
            os.path.join("scripts", "make_control_figures.py"),
        )
    )
    subprocess.call(["python", script_path, work_dir])


def read(work_dir, style="dict"):
    """
    Reads work_dir/result/{site:s}_{particle:s}.csv into dict[site][particle].
    """
    CFG = read_config(work_dir, ["sites", "particles",])
    mag = {}
    for skey in CFG["sites"]:
        mag[skey] = {}
        for pkey in CFG["particles"]:
            path = wdir.join(work_dir, "result", "{:s}_{:s}.csv".format(skey, pkey),)
            df = pandas.read_csv(path)
            if style == "record":
                mag[skey][pkey] = df.to_records()
            elif style == "dict":
                mag[skey][pkey] = df.to_dict("list")
            else:
                raise KeyError("Unknown style: '{:s}'.".format(style))
    return mag


def read_config(work_dir, keys=["sites", "particles", "pointing", "config", "plotting"]):
    CFG = {}
    for key in keys:
        CFG[key] = tools.read_json(wdir.join(work_dir, "config", key + ".json"))
    return CFG


def read_statistics_site_particle(map_site_particle_dir):
    map_dir = map_site_particle_dir
    paths = glob.glob(os.path.join(map_dir, "*_statistics.recarray.tar"))
    basenames = [os.path.basename(p) for p in paths]
    job_ids = [int(b[0:6]) for b in basenames]
    job_ids.sort()

    stats = Records.init(
        dtypes={
            "particle_azimuth_deg": "f4",
            "particle_zenith_deg": "f4",
            "particle_energy_GeV": "f4",
            "num_photons": "f4",
            "num_bunches": "i4",
            "position_med_x_m": "f4",
            "position_med_y_m": "f4",
            "position_phi_rad": "f4",
            "position_std_minor_m": "f4",
            "position_std_major_m": "f4",
            "direction_med_cx_rad": "f4",
            "direction_med_cy_rad": "f4",
            "direction_phi_rad": "f4",
            "direction_std_minor_rad": "f4",
            "direction_std_major_rad": "f4",
            "arrival_time_mean_s": "f4",
            "arrival_time_median_s": "f4",
            "arrival_time_std_s": "f4",
            "off_axis_deg": "f4",
        }
    )

    num_jobs = len(job_ids)
    for i, job_id in enumerate(job_ids):
        print("Read job: {:06d}, {: 6d} / {: 6d}".format(job_id, i, num_jobs))
        s_path = os.path.join(
            map_dir, "{:06d}_statistics.recarray.tar".format(job_id)
        )
        pools = recarray_io.read_from_tar(path=s_path)
        stats = Records.append_numpy_recarray(stats, pools)

    return Records.to_numpy_recarray(stats)
