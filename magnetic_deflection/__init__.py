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

import os
import json_numpy
import pandas
import numpy as np
import scipy
from scipy.optimize import curve_fit as scipy_optimize_curve_fit
import shutil
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
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, "sites.json"), "wt") as f:
        f.write(json_numpy.dumps(sites, indent=4))
    with open(os.path.join(work_dir, "pointing.json"), "wt") as f:
        f.write(json_numpy.dumps(plenoscope_pointing, indent=4))
    with open(os.path.join(work_dir, "particles.json"), "wt") as f:
        f.write(json_numpy.dumps(particles, indent=4))
    _write_default_config(
        work_dir=work_dir,
        energy_supports_max=max_energy,
        energy_supports_num=num_energy_supports
    )
    # write default plotting config
    with open(os.path.join(work_dir, "plotting.json"), "wt") as f:
        f.write(json_numpy.dumps(examples.PLOTTING, indent=4))


def _write_default_config(work_dir, energy_supports_max, energy_supports_num):
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
    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(json_numpy.dumps(cfg, indent=4))


def B_make_jobs_from_work_dir(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    pointing = tools.read_json(os.path.join(work_dir, "pointing.json"))
    config = tools.read_json(os.path.join(work_dir, "config.json"))

    return map_and_reduce.make_jobs(
        work_dir=work_dir,
        sites=sites,
        particles=particles,
        pointing=pointing,
        **config,
    )


def B2_read_job_results_from_work_dir(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    map_dir = os.path.join(work_dir, "map")

    job_results = {}
    for skey in sites:
        job_results[skey] = {}
        for pkey in particles:
            paths = glob.glob(
                os.path.join(map_dir, skey, pkey, "*_result.json")
            )
            job_results[skey][pkey] = [
                json_numpy.loads(open(p, "rt").read()) for p in paths
            ]

    return job_results


def C_reduce_job_results_in_work_dir(job_results, work_dir):
    raw_deflection_table_path = os.path.join(work_dir, "raw")
    os.makedirs(raw_deflection_table_path, exist_ok=True)

    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))

    table = {}
    for skey in sites:
        table[skey] = {}
        for pkey in particles:
            df = pandas.DataFrame(job_results[skey][pkey])
            rec = df.to_records(index=False)
            order = np.argsort(rec["particle_energy_GeV"])
            rec = rec[order]
            table[skey][pkey] = rec

    tools.write_deflection_table(
        deflection_table=table, path=raw_deflection_table_path
    )


def C2_reduce_statistics_in_work_dir(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    shower_statistics_dir = os.path.join(work_dir, "shower_statistics")

    os.makedirs(shower_statistics_dir, exist_ok=True)

    for skey in sites:
        for pkey in particles:
            print("Reducing shower statistics: ", skey, pkey)

            sp = tools.read_statistics_site_particle(
                map_site_particle_dir=os.path.join(work_dir, "map", skey, pkey)
            )

            sp_dir = os.path.join(shower_statistics_dir, skey, pkey)
            os.makedirs(sp_dir, exist_ok=True)
            recarray_io.write_to_tar(
                recarray=sp,
                path=os.path.join(sp_dir, "shower_statistics.tar")
            )


def read_shower_statistics(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    shower_statistics_dir = os.path.join(work_dir, "shower_statistics")
    out = {}
    for skey in sites:
        out[skey] = {}
        for pkey in particles:
            out[skey][pkey] = recarray_io.read_from_tar(path=os.path.join(
                shower_statistics_dir, skey, pkey, "shower_statistics.tar"))
    return out


def D_summarize_raw_deflection(
    work_dir, min_fit_energy=0.65,
):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    pointing = tools.read_json(os.path.join(work_dir, "pointing.json"))
    min_particle_energy = np.min(
        [np.min(particles[p]["energy_bin_edges_GeV"]) for p in particles]
    )
    if min_particle_energy > min_fit_energy:
        min_fit_energy = 1.1 * min_particle_energy
    _cut_invalid(
        in_path=os.path.join(work_dir, "raw"),
        out_path=os.path.join(work_dir, "raw_valid"),
        min_energy=min_fit_energy,
    )
    _add_density_fields(
        in_path=os.path.join(work_dir, "raw_valid"),
        out_path=os.path.join(work_dir, "raw_valid_add"),
    )
    _smooth_and_reject_outliers(
        in_path=os.path.join(work_dir, "raw_valid_add"),
        out_path=os.path.join(work_dir, "raw_valid_add_clean"),
    )
    _set_high_energies(
        particles=particles,
        in_path=os.path.join(work_dir, "raw_valid_add_clean"),
        out_path=os.path.join(work_dir, "raw_valid_add_clean_high"),
    )
    sites2 = {}
    for skey in sites:
        if "Off" not in skey:
            sites2[skey] = sites[skey]
    _fit_power_law(
        particles=particles,
        sites=sites2,
        in_path=os.path.join(work_dir, "raw_valid_add_clean_high"),
        out_path=os.path.join(work_dir, "raw_valid_add_clean_high_power"),
    )
    _export_table(
        particles=particles,
        sites=sites2,
        in_path=os.path.join(work_dir, "raw_valid_add_clean_high_power"),
        out_path=os.path.join(work_dir, "result"),
    )

    script_path = os.path.abspath(
        pkg_resources.resource_filename(
            "magnetic_deflection",
            os.path.join("scripts", "make_control_figures.py"),
        )
    )
    subprocess.call(["python", script_path, work_dir])


def Z_get_incomplete_jobs(work_dir):
    jobs_path_state = Z_get_incomplete_job_paths(work_dir)
    jobs = []
    for job_path_state in jobs_path_state:
        job_path, job_state = job_path_state
        print("job", job_path, "state:", job_state)
        job = tools.read_json(job_path)
        jobs.append(job)
    return jobs


def Z_get_incomplete_job_paths(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))

    incomplete_ids = []
    for skey in sites:
        for pkey in particles:
            map_dir = os.path.join(work_dir, "map", skey, pkey)
            log_wildcard = os.path.join(map_dir, "*_log.jsonl")
            log_paths = glob.glob(log_wildcard)
            if len(log_paths) == 0:
                RuntimeWarning("Can't glob any log_paths.")

            log_basenames = [os.path.basename(p) for p in log_paths]
            job_ids = [int(p[0:6]) for p in log_basenames]

            for job_id in job_ids:
                log_path = os.path.join(map_dir, "{:06d}_log.jsonl".format(job_id))
                job_path = os.path.join(map_dir, "{:06d}_job.json".format(job_id))
                joblog = tools.read_jsonl(log_path)
                if len(joblog) > 0:
                    last_joblog = joblog[-1]
                    if "m" in last_joblog:
                        if last_joblog["m"] != "job: end":
                            incomplete_ids.append((job_path, "no end"))
                    else:
                        incomplete_ids.append((job_path, "no 'm'"))

                else:
                    incomplete_ids.append((job_path, "log empty"))

            not_even_log_ids = _get_incomplete_job_paths(
                started_wildcard=os.path.join(map_dir, "*_job.json"),
                completed_wildcard=os.path.join(map_dir, "*_log.jsonl"),
            )
            for not_even_log_id in not_even_log_ids:
                job_path = os.path.join(map_dir, "{:06d}_job.json".format(not_even_log_id))
                incomplete_ids.append((job_path, "no log"))

    return incomplete_ids


def _get_incomplete_job_paths(started_wildcard, completed_wildcard):
    started_paths = glob.glob(started_wildcard)
    completed_paths = glob.glob(completed_wildcard)
    if len(started_paths) == 0:
        RuntimeWarning("Can't glob any started_paths.")

    started_basenames = [os.path.basename(p) for p in started_paths]
    completed_basenames = [os.path.basename(p) for p in completed_paths]

    started_ids = [int(p[0:6]) for p in started_basenames]
    completed_ids = [int(p[0:6]) for p in completed_basenames]

    completed_ids = set(completed_ids)

    incomplete_ids = []
    for started_id in started_ids:
        if started_id not in completed_ids:
            incomplete_ids.append(started_id)
    return incomplete_ids


def _cut_invalid(
    in_path, out_path, min_energy,
):
    os.makedirs(out_path, exist_ok=True)
    raw_deflection_table = tools.read_deflection_table(path=in_path)
    deflection_table = analysis.cut_invalid_from_deflection_table(
        deflection_table=raw_deflection_table, min_energy=min_energy
    )
    tools.write_deflection_table(
        deflection_table=deflection_table, path=out_path
    )


def _add_density_fields(
    in_path, out_path,
):
    os.makedirs(out_path, exist_ok=True)
    valid_deflection_table = tools.read_deflection_table(path=in_path)
    deflection_table = analysis.add_density_fields_to_deflection_table(
        deflection_table=valid_deflection_table
    )
    tools.write_deflection_table(
        deflection_table=deflection_table, path=out_path
    )


FIT_KEYS = {
    "particle_azimuth_deg": {"start": 90.0,},
    "particle_zenith_deg": {"start": 0.0,},
    "position_med_x_m": {"start": 0.0,},
    "position_med_y_m": {"start": 0.0,},
}


def _smooth_and_reject_outliers(in_path, out_path):
    deflection_table = tools.read_deflection_table(path=in_path)
    smooth_table = {}
    for skey in deflection_table:
        smooth_table[skey] = {}
        for pkey in deflection_table[skey]:
            t = deflection_table[skey][pkey]
            sm = {}
            for key in FIT_KEYS:
                sres = analysis.smooth(
                    energies=t["particle_energy_GeV"], values=t[key]
                )
                if "particle_energy_GeV" in sm:
                    np.testing.assert_array_almost_equal(
                        x=sm["particle_energy_GeV"],
                        y=sres["energy_supports"],
                        decimal=3,
                    )
                else:
                    sm["particle_energy_GeV"] = sres["energy_supports"]
                sm[key] = sres["key_mean80"]
                sm[key + "_std"] = sres["key_std80"]
                df = pandas.DataFrame(sm)
            smooth_table[skey][pkey] = df.to_records(index=False)
    os.makedirs(out_path, exist_ok=True)
    tools.write_deflection_table(deflection_table=smooth_table, path=out_path)


def _set_high_energies(
    particles,
    in_path,
    out_path,
    energy_start=200,
    energy_stop=600,
    num_points=20,
):
    deflection_table = tools.read_deflection_table(path=in_path)

    charge_signs = {}
    for pkey in particles:
        charge_signs[pkey] = np.sign(particles[pkey]["electric_charge_qe"])

    out = {}
    for skey in deflection_table:
        out[skey] = {}
        for pkey in deflection_table[skey]:
            t = deflection_table[skey][pkey]
            sm = {}
            for key in FIT_KEYS:
                sm["particle_energy_GeV"] = np.array(
                    t["particle_energy_GeV"].tolist()
                    + np.geomspace(
                        energy_start, energy_stop, num_points
                    ).tolist()
                )
                key_start = charge_signs[pkey] * FIT_KEYS[key]["start"]
                sm[key] = np.array(
                    t[key].tolist()
                    + (key_start * np.ones(num_points)).tolist()
                )
            df = pandas.DataFrame(sm)
            df = df[df["particle_zenith_deg"] <= corsika.MAX_ZENITH_DEG]
            out[skey][pkey] = df.to_records(index=False)
    os.makedirs(out_path, exist_ok=True)
    tools.write_deflection_table(deflection_table=out, path=out_path)


def _fit_power_law(
    particles, sites, in_path, out_path,
):
    deflection_table = tools.read_deflection_table(path=in_path)
    charge_signs = {}
    for pkey in particles:
        charge_signs[pkey] = np.sign(particles[pkey]["electric_charge_qe"])
    os.makedirs(out_path, exist_ok=True)
    for skey in sites:
        for pkey in particles:
            t = deflection_table[skey][pkey]
            fits = {}
            for key in FIT_KEYS:
                key_start = charge_signs[pkey] * FIT_KEYS[key]["start"]
                if np.mean(t[key] - key_start) > 0:
                    sig = -1
                else:
                    sig = 1

                try:
                    expy, _ = scipy_optimize_curve_fit(
                        analysis.power_law,
                        t["particle_energy_GeV"],
                        t[key] - key_start,
                        p0=(sig * charge_signs[pkey], 1.0),
                    )
                except RuntimeError as err:
                    print(err)
                    expy = [0.0, 0.0]

                fits[key] = {
                    "formula": "f(energy) = scale*energy**index + offset",
                    "scale": float(expy[0]),
                    "index": float(expy[1]),
                    "offset": float(key_start),
                }
            filename = "{:s}_{:s}".format(skey, pkey)
            filepath = os.path.join(out_path, filename)
            with open(filepath + ".json", "wt") as fout:
                fout.write(json_numpy.dumps(fits, indent=4))


def _export_table(
    particles, sites, in_path, out_path,
):
    os.makedirs(out_path, exist_ok=True)
    for skey in sites:
        for pkey in particles:
            out = {}
            out["particle_energy_GeV"] = np.geomspace(
                np.min(particles[pkey]["energy_bin_edges_GeV"]),
                np.max(particles[pkey]["energy_bin_edges_GeV"]),
                1024,
            )
            filename = "{:s}_{:s}".format(skey, pkey)
            filepath = os.path.join(in_path, filename)
            with open(filepath + ".json", "rt") as fin:
                power_law = json_numpy.loads(fin.read())
            for key in FIT_KEYS:
                rec_key = analysis.power_law(
                    energy=out["particle_energy_GeV"],
                    scale=power_law[key]["scale"],
                    index=power_law[key]["index"],
                )
                rec_key += power_law[key]["offset"]
                out[key] = rec_key
            df = pandas.DataFrame(out)
            df = df[df["particle_zenith_deg"] <= corsika.MAX_ZENITH_DEG]
            csv = df.to_csv(index=False)
            out_filepath = os.path.join(out_path, filename)
            with open(out_filepath + ".csv.tmp", "wt") as fout:
                fout.write(csv)
            shutil.move(out_filepath + ".csv.tmp", out_filepath + ".csv")


def read(work_dir, style="dict"):
    """
    Reads work_dir/result/{site:s}_{particle:s}.csv into dict[site][particle].
    """
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    mag = {}
    for skey in sites:
        mag[skey] = {}
        for pkey in particles:
            path = os.path.join(
                work_dir, "result", "{:s}_{:s}.csv".format(skey, pkey),
            )
            df = pandas.read_csv(path)
            if style == "record":
                mag[skey][pkey] = df.to_records()
            elif style == "dict":
                mag[skey][pkey] = df.to_dict("list")
            else:
                raise KeyError("Unknown style: '{:s}'.".format(style))
    return mag


def read_config(work_dir):
    cf = {}
    cf["sites"] = tools.read_json(os.path.join(work_dir, "sites.json"))
    cf["particles"] = tools.read_json(os.path.join(work_dir, "particles.json"))
    cf["pointing"] = tools.read_json(os.path.join(work_dir, "pointing.json"))
    cf["config"] = tools.read_json(os.path.join(work_dir, "config.json"))
    cf["plotting"] = tools.read_json(os.path.join(work_dir, "plotting.json"))
    return cf
