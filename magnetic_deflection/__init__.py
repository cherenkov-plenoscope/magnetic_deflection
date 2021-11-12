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
    os.makedirs(work_dir, exist_ok=True)

    tools.write_json(os.path.join(work_dir, "sites.json"), sites)
    tools.write_json(os.path.join(work_dir, "pointing.json"), plenoscope_pointing)
    tools.write_json(os.path.join(work_dir, "particles.json"), particles)

    _write_default_config(
        work_dir=work_dir,
        energy_supports_max=max_energy,
        energy_supports_num=num_energy_supports
    )
    _write_default_plotting_config(work_dir=work_dir)


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
    tools.write_json(path=os.path.join(work_dir, "config.json"), obj=cfg)

def _write_default_plotting_config(work_dir):
    tools.write_json(
        path=os.path.join(work_dir, "plotting.json"),
        obj=examples.PLOTTING,
    )


def B_make_jobs_from_work_dir(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    pointing = tools.read_json(os.path.join(work_dir, "pointing.json"))
    config = tools.read_json(os.path.join(work_dir, "config.json"))

    jobs = []
    job_id = 0
    for skey in sites:
        for pkey in particles:
            print("Make jobs ", skey, pkey)
            site_particle_jobs = map_and_reduce.make_jobs(
                first_job_id=job_id,
                work_dir=work_dir,
                site=sites[skey],
                site_key=skey,
                particle=particles[pkey],
                particle_key=pkey,
                pointing=pointing,
                energy_supports_min=min(particles[pkey]["energy_bin_edges_GeV"]),
                **config,
            )
            job_id += len(site_particle_jobs)
            jobs += site_particle_jobs

    print("Sort jobs by energy")
    return tools.sort_records_by_key(
        records=jobs, keys=("particle", "energy_GeV")
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
            job_results[skey][pkey] = [tools.read_json(p) for p in paths]

    return job_results


def C_reduce_job_results_in_work_dir(job_results, work_dir):
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))

    table = {}
    for skey in sites:
        table[skey] = {}
        for pkey in particles:
            print("Reducing shower results: ", skey, pkey)

            df = pandas.DataFrame(job_results[skey][pkey])
            rec = df.to_records(index=False)
            order = np.argsort(rec["particle_energy_GeV"])
            rec = rec[order]
            table[skey][pkey] = rec

            recarray_io.write_to_csv(
                recarray=table[skey][pkey],
                path=os.path.join(raw_dir, "{:s}_{:s}.csv".format(skey, pkey))
            )


def C2_reduce_statistics_in_work_dir(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    ss_dir = os.path.join(work_dir, "shower_statistics")

    os.makedirs(ss_dir, exist_ok=True)

    for skey in sites:
        for pkey in particles:
            print("Reducing shower statistics: ", skey, pkey)

            sp = tools.read_statistics_site_particle(
                map_site_particle_dir=os.path.join(work_dir, "map", skey, pkey)
            )

            ss_s_p_dir = os.path.join(ss_dir, skey, pkey)
            os.makedirs(sp_dir, exist_ok=True)
            recarray_io.write_to_tar(
                recarray=sp,
                path=os.path.join(ss_s_p_dir, "shower_statistics.tar")
            )


def read_shower_statistics(work_dir):
    sites = tools.read_json(os.path.join(work_dir, "sites.json"))
    particles = tools.read_json(os.path.join(work_dir, "particles.json"))
    ss_dir = os.path.join(work_dir, "shower_statistics")
    out = {}
    for skey in sites:
        out[skey] = {}
        for pkey in particles:
            out[skey][pkey] = recarray_io.read_from_tar(
                path=os.path.join(ss_dir, skey, pkey, "shower_statistics.tar")
            )
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

    raw_dir = os.path.join(work_dir, "raw")

    raw_valid_dir = os.path.join(work_dir, "raw_valid")
    os.makedirs(raw_valid_dir, exist_ok=True)

    raw_valid_add_dir = os.path.join(work_dir, "raw_valid_add")
    os.makedirs(raw_valid_add_dir, exist_ok=True)

    raw_valid_add_clean_dir = os.path.join(work_dir, "raw_valid_add_clean")
    os.makedirs(raw_valid_add_clean_dir, exist_ok=True)

    raw_valid_add_clean_high_dir = os.path.join(work_dir, "raw_valid_add_clean_high")
    os.makedirs(raw_valid_add_clean_high_dir, exist_ok=True)

    raw_valid_add_clean_high_power_dir = os.path.join(work_dir, "raw_valid_add_clean_high_power")
    os.makedirs(raw_valid_add_clean_high_power_dir, exist_ok=True)

    result_dir = os.path.join(work_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    for skey in sites:
        for pkey in particles:


            fname = skey + "_" + pkey + ".csv"
            charge_sign = np.sign(particles[pkey]["electric_charge_qe"])

            print(skey, pkey, fname)

            raw = recarray_io.read_from_csv(path=os.path.join(raw_dir, fname))

            # cut invalid
            # -----------
            raw_valid = analysis.cut_invalid_from_deflection(
                deflection=raw,
                min_energy=min_fit_energy,
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
                path=os.path.join(raw_valid_add_clean_dir, fname)
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
                path=os.path.join(raw_valid_add_clean_high_dir, fname)
            )

            # fit_power_law
            # -------------
            power_law_fit = analysis.fit_power_law_to_deflection(
                deflection=raw_valid_add_clean_high,
                charge_sign=charge_sign,
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
