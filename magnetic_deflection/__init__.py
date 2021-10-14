from . import discovery
from . import map_and_reduce
from . import examples
from . import analysis
from . import light_field_characterization
from . import corsika
from . import spherical_coordinates
from . import tools
from . import jsonl_logger

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
import pickle


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
    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(
            json_numpy.dumps(
                {
                    "max_energy_GeV": float(max_energy),
                    "num_energy_supports": int(num_energy_supports),
                },
                indent=4,
            )
        )


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
        energy_supports_max=config["max_energy_GeV"],
        energy_supports_num=config["num_energy_supports"],
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
