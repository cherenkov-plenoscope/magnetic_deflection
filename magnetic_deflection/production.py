import atmospheric_cherenkov_response as acr
from . import allsky
import os


def init(work_dir, corsika_primary_path):
    os.makedirs(work_dir, exist_ok=True)

    for sk in acr.sites.keys():
        sk_dir = os.path.join(work_dir, sk)
        for pk in acr.particles.keys():
            sk_pk_dir = os.path.join(sk_dir, pk)

            if not os.path.exists(sk_pk_dir):
                particle = acr.particles.init(pk)

                energy_start_GeV = acr.particles.compile_energy(
                    particle["population"]["energy"]["start_GeV"]
                )

                allsky.init(
                    work_dir=sk_pk_dir,
                    particle_key=pk,
                    site_key=sk,
                    energy_start_GeV=energy_start_GeV,
                    energy_stop_GeV=64.0,
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
