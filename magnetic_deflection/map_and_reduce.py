import pandas as pd
import shutil
import os
import glob
import numpy as np
import corsika_primary_wrapper as cpw
from . import examples
from . import discovery
from . import light_field_characterization


def make_jobs(
    sites,
    particles,
    plenoscope_pointing,
    max_energy,
    num_energy_supports,
    energy_supports_power_law_slope=-1.7,
    iteration_speed=0.9,
    initial_num_events_per_iteration=2**5,
    max_total_num_events=2**12,
    outlier_percentile=50.0,
    corsika_primary_path=examples.CORSIKA_PRIMARY_MOD_PATH,
):
    jobs = []
    for site_key in sites:
        for particle_key in particles:
            site = sites[site_key]
            particle_id = particles[particle_key]["particle_id"]
            min_energy = np.min(particles[particle_key][
                "energy_bin_edges_GeV"])
            energy_supports = powerspace(
                start=min_energy,
                stop=max_energy,
                power_index=energy_supports_power_law_slope,
                num=num_energy_supports)
            for energy_idx in range(len(energy_supports)):
                job = {}
                job['site'] = site
                job['primary_energy'] = energy_supports[energy_idx]
                job['primary_particle_id'] = particle_id
                job['instrument_azimuth_deg'] = plenoscope_pointing[
                    'azimuth_deg']
                job['instrument_zenith_deg'] = plenoscope_pointing[
                    'zenith_deg']
                job['max_off_axis_deg'] = particles[particle_key][
                    "magnetic_deflection_max_off_axis_deg"]
                job['corsika_primary_path'] = corsika_primary_path
                job['site_key'] = site_key
                job['particle_key'] = particle_key
                job['iteration_speed'] = iteration_speed
                job['initial_num_events_per_iteration'] = (
                    initial_num_events_per_iteration)
                job['max_total_num_events'] = max_total_num_events
                job['outlier_percentile'] = outlier_percentile
                jobs.append(job)
    return sort_jobs_by_key(jobs=jobs, key='primary_energy')


def sort_jobs_by_key(jobs, key):
    _values = [job[key] for job in jobs]
    _values_argsort = np.argsort(_values)
    jobs_sorted = [jobs[_values_argsort[i]] for i in range(len(jobs))]
    return jobs_sorted


def run_job(job):
    deflection = discovery.estimate_deflection(
        site=job['site'],
        primary_energy=job['primary_energy'],
        primary_particle_id=job['primary_particle_id'],
        instrument_azimuth_deg=job['instrument_azimuth_deg'],
        instrument_zenith_deg=job['instrument_zenith_deg'],
        max_off_axis_deg=job['max_off_axis_deg'],
        initial_num_events_per_iteration=job[
            'initial_num_events_per_iteration'],
        max_total_num_events=job['max_total_num_events'],
        corsika_primary_path=job['corsika_primary_path'],
        iteration_speed=job['iteration_speed'],
    )
    deflection['particle_id'] = job['primary_particle_id']
    deflection['energy_GeV'] = job['primary_energy']
    deflection['site_key'] = job['site_key']
    deflection['particle_key'] = job['particle_key']

    lfc = light_field_characterization.characterize_cherenkov_pool(
        site=job['site'],
        primary_energy=job['primary_energy'],
        primary_particle_id=job['primary_particle_id'],
        primary_azimuth_deg=deflection['primary_azimuth_deg'],
        primary_zenith_deg=deflection['primary_zenith_deg'],
        corsika_primary_path=job['corsika_primary_path'],
        total_energy_thrown=1e2,
        min_num_cherenkov_photons=1e2,
        outlier_percentile=job['outlier_percentile'])
    deflection.update(lfc)

    return deflection


KEEP_KEYS = [
    "particle_id",
    "energy_GeV",
    "primary_azimuth_deg",
    "primary_zenith_deg",
    "cherenkov_pool_x_m",
    "cherenkov_pool_y_m",
    "off_axis_deg",
    "num_valid_Cherenkov_pools",
    "num_thrown_Cherenkov_pools",
    "total_num_events",
]


def structure_combined_results(
    combined_results,
    particles,
    sites,
):
    df = pd.DataFrame(combined_results)

    all_keys_keep = KEEP_KEYS + light_field_characterization.KEYS

    res = {}
    for site_key in sites:
        res[site_key] = {}
        for particle_key in particles:
            site_mask = (df['site_key'] == site_key).values
            particle_mask = (df['particle_key'] == particle_key).values
            mask = np.logical_and(site_mask, particle_mask)
            site_particle_df = df[mask]
            site_particle_df = site_particle_df[site_particle_df['valid']]
            site_particle_keep_df = site_particle_df[all_keys_keep]
            site_particle_rec = site_particle_keep_df.to_records(index=False)
            argsort = np.argsort(site_particle_rec['energy_GeV'])
            site_particle_rec = site_particle_rec[argsort]
            res[site_key][particle_key] = site_particle_rec
    return res


def write_recarray_to_csv(recarray, path):
    df = pd.DataFrame(recarray)
    csv = df.to_csv(index=False)
    with open(path+".tmp", 'wt') as f:
        f.write(csv)
    shutil.move(path+".tmp", path)


def read_csv_to_recarray(path):
    df = pd.read_csv(path)
    rec = df.to_records(index=False)
    return rec


def read_deflection_table(path):
    paths = glob.glob(os.path.join(path, "*.csv"))
    deflection_table = {}
    for pa in paths:
        basename = os.path.basename(pa)
        name = basename.split('.')[0]
        split_name = name.split('_')
        assert len(split_name) == 2
        site_key, particle_key = split_name
        if site_key not in deflection_table:
            deflection_table[site_key] = {}
        deflection_table[site_key][particle_key] = read_csv_to_recarray(pa)
    return deflection_table


def write_deflection_table(deflection_table, path):
    for site_key in deflection_table:
        for particle_key in deflection_table[site_key]:
            out_path = os.path.join(path, '{:s}_{:s}.csv'.format(
                site_key, particle_key))
            write_recarray_to_csv(
                recarray=deflection_table[site_key][particle_key],
                path=out_path)


def powerspace(start, stop, power_index, num, iterations=10000):
    assert num >= 2
    num_points_without_start_and_end = num - 2
    if num_points_without_start_and_end >= 1:
        full = []
        for iti in range(iterations):
            points = np.sort(cpw.random_distributions.draw_power_law(
                lower_limit=start,
                upper_limit=stop,
                power_slope=power_index,
                num_samples=num_points_without_start_and_end))
            points = [start] + points.tolist() + [stop]
            full.append(points)
        full = np.array(full)
        return np.mean(full, axis=0)
    else:
        return np.array([start, stop])


def read_csv_to_dict(path):
    return pd.read_csv(path).to_dict(orient='list')
