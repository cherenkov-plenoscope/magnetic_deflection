import atmospheric_cherenkov_response as acr
import magnetic_deflection as mdfl
import binning_utils as bu
import os
import multiprocessing

pool = multiprocessing.Pool(6)


work_dir = "sky"

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

            mdfl.allsky.init(
                work_dir=sk_pk_dir,
                particle_key=pk,
                site_key=sk,
                energy_start_GeV=energy_start_GeV,
                energy_stop_GeV=64.0,
                corsika_primary_path=os.path.join(
                    "/",
                    "home",
                    "relleums",
                    "Desktop",
                    "starter_kit",
                    "build",
                    "corsika",
                    "modified",
                    "corsika-75600",
                    "run",
                    "corsika75600Linux_QGSII_urqmd",
                ),
            )


for trip in range(10):
    for sk in acr.sites.keys():
        sk_dir = os.path.join(work_dir, sk)
        for pk in acr.particles.keys():
            sk_pk_dir = os.path.join(sk_dir, pk)

            print(sk, pk)
            allsky = mdfl.allsky.open(sk_pk_dir)
            allsky.populate(
                pool=pool,
                num_chunks=1,
                num_jobs=6,
                num_showers_per_job=1000,
            )
