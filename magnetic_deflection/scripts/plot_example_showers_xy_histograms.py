#!/usr/bin/python
import sys
import os
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "plot_example_showers_xy_histograms")
os.makedirs(out_dir, exist_ok=True)

CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]

matplotlib.rcParams["mathtext.fontset"] = PLT["rcParams"]["mathtext.fontset"]
matplotlib.rcParams["font.family"] = PLT["rcParams"]["font.family"]

prng = np.random.Generator(np.random.PCG64(seed=1))

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 1.5}
CMAP_FIGSIZE = {"rows": 240, "cols": 1280, "fontsize": 1.5}

pkey = "helium"

cases = {
    "low_energy": {
        "energy": {"start": 10, "stop": 16},
        "area": {"start": 1e5, "stop": 5e5},
    },
    "high_energy": {
        "energy": {"start": 32, "stop": 64},
        "area": {"start": 2e6, "stop": 1e7},
    },
}

XY_BIN_EDGES = np.linspace(-1e4, 1e4, 201)
ON_AXIS_SCALE = 1.0
MAX_NUM_EXAMPLES = 8

# find example events
# -------------------
shower_statistics = mdfl.read_statistics(work_dir=work_dir)
shower_explicit_steerings = mdfl.read_explicit_steerings(work_dir=work_dir)

example_events = {}

for skey in CFG["sites"]:
    example_events[skey] = {}
    for ckey in cases:
        example_events[skey][ckey] = {}
        sp_stats = shower_statistics[skey][pkey]
        E_cut = cases[ckey]["energy"]
        A_cut = cases[ckey]["area"]

        for i in range(len(sp_stats)):
            E = sp_stats["particle_energy_GeV"][i]
            A = (
                sp_stats["position_std_major_m"][i]
                * sp_stats["position_std_minor_m"][i]
                * np.pi
            )
            e = E_cut["start"] <= E < E_cut["stop"]
            a = A_cut["start"] <= A < A_cut["stop"]
            on_axis = (
                sp_stats["off_axis_deg"][i]
                <= ON_AXIS_SCALE
                * CFG["particles"][pkey][
                    "magnetic_deflection_max_off_axis_deg"
                ]
            )

            if on_axis and e and a:
                evtkey = (sp_stats["run"][i], sp_stats["event"][i])
                example_events[skey][ckey][evtkey] = sp_stats[i : i + 1]


# limit number example events
# ---------------------------
for skey in CFG["sites"]:
    for ckey in cases:
        all_evtkeys = list(example_events[skey][ckey].keys())
        num_examples = len(all_evtkeys)
        if num_examples > MAX_NUM_EXAMPLES:
            choice = prng.choice(
                a=num_examples, size=MAX_NUM_EXAMPLES, replace=False
            )
            out = {}
            for c in choice:
                evtkey = all_evtkeys[c]
                out[evtkey] = example_events[skey][ckey][evtkey]
            example_events[skey][ckey] = out


def get_event_id_of_first_event_in_run(steering_card):
    for line in str.splitlines(steering_card):
        if "EVTNR" in line:
            val = str.replace(line, "EVTNR", "")
            val = str.strip(val)
            return int(val)
    raise ValueError("steering_card has no 'EVTNR'.")


def replace_EVTNR_in_steering_card(steering_card, evtnr):
    assert evtnr > 0
    replaced = False
    out = ""
    for line in str.splitlines(steering_card):
        if "EVTNR" in line:
            out += "EVTNR" + " " + "{:d}".format(evtnr) + "\n"
            replaced = True
        else:
            out += line + "\n"
    if not replaced:
        raise ValueError("steering_card has no 'EVTNR'")
    return out


def replace_NSHOW_in_steering_card(steering_card, nshow):
    assert nshow > 0
    replaced = False
    out = ""
    for line in str.splitlines(steering_card):
        if "NSHOW" in line:
            out += "NSHOW" + " " + "{:d}".format(nshow) + "\n"
            replaced = True
        else:
            out += line + "\n"
    if not replaced:
        raise ValueError("steering_card has no 'NSHOW'")
    return out


# get explicit steering
# ---------------------
example_steerings = {}
for skey in CFG["sites"]:
    example_steerings[skey] = {}
    for ckey in cases:
        example_steerings[skey][ckey] = {}
        for evtkey in example_events[skey][ckey]:
            run_id = evtkey[0]
            event_id = evtkey[1]

            steering = {}
            steering["steering_card"] = str(
                shower_explicit_steerings[skey][pkey][run_id]["steering_card"]
            )
            first_event_id_in_run = get_event_id_of_first_event_in_run(
                steering_card=steering["steering_card"]
            )
            event_idx = event_id - first_event_id_in_run
            assert event_idx >= 0
            steering["primary_bytes"] = bytes(
                mdfl.corsika.cpw._primaries_slice(
                    primary_bytes=shower_explicit_steerings[skey][pkey][run_id][
                        "primary_bytes"
                    ],
                    i=event_idx,
                )
            )
            evtkey = (run_id, event_id)
            example_steerings[skey][ckey][evtkey] = steering

            primary_dict = mdfl.corsika.cpw._primaries_to_dict(
                primary_bytes=steering["primary_bytes"]
            )[0]
            assert (
                np.abs(
                    primary_dict["energy_GeV"]
                    - example_events[skey][ckey][evtkey]["particle_energy_GeV"]
                )
                < 1e-3
            )
            assert (
                np.abs(
                    primary_dict["zenith_rad"]
                    - np.deg2rad(
                        example_events[skey][ckey][evtkey][
                            "particle_zenith_deg"
                        ]
                    )
                )
                < 1e-3
            )
            assert (
                np.abs(
                    mdfl.spherical_coordinates._azimuth_range(
                        np.rad2deg(primary_dict["azimuth_rad"])
                    )
                    - example_events[skey][ckey][evtkey][
                        "particle_azimuth_deg"
                    ]
                )
                < 1e-3
            )


# reproduce cherenkov pools
# -------------------------
jobs = []

for skey in CFG["sites"]:
    for ckey in cases:
        for evtkey in example_steerings[skey][ckey]:
            run_id = evtkey[0]
            event_id = evtkey[1]

            steer = example_steerings[skey][ckey][evtkey]
            job = {}
            job["steering_card"] = str(steer["steering_card"])
            job["steering_card"] = replace_EVTNR_in_steering_card(
                steering_card=job["steering_card"], evtnr=event_id,
            )
            job["steering_card"] = replace_NSHOW_in_steering_card(
                steering_card=job["steering_card"], nshow=1,
            )
            job["primary_bytes"] = bytes(steer["primary_bytes"])
            job["skey"] = skey
            job["pkey"] = pkey
            job["ckey"] = ckey
            job["run"] = run_id
            job["event"] = event_id
            job["corsika_primary_path"] = CFG["config"]["corsika_primary_path"]
            job["xy_bin_edges"] = XY_BIN_EDGES
            job["shower_statistic"] = example_events[skey][ckey][evtkey]
            job["density_cut"] = {
                "median": {"percentile": 50.0},
            }  # CFG["config"]["density_cut"]
            assert job["shower_statistic"]["run"] == job["run"]
            assert job["shower_statistic"]["event"] == job["event"]
            jobs.append(job)


def run_job(job):
    job_dir = os.path.join(
        out_dir, "cherenkov_pools", job["skey"], job["pkey"], job["ckey"],
    )
    os.makedirs(job_dir, exist_ok=True)
    job_key = "{:06d}_{:06d}".format(job["run"], job["event"])

    pool_path = os.path.join(job_dir, job_key + "_cherenkov_pool.tar")
    hist_path = os.path.join(
        job_dir, job_key + "_cherenkov_pool_histogram_xy.jpg"
    )

    # reproduce Cherenkov-pool
    # ------------------------
    if not os.path.exists(pool_path):
        mdfl.corsika.cpw.explicit_corsika_primary(
            corsika_path=job["corsika_primary_path"],
            steering_card=job["steering_card"],
            primary_bytes=job["primary_bytes"],
            output_path=pool_path,
            stdout_postfix=".stdout",
            stderr_postfix=".stderr",
            tmp_dir_prefix="corsika_primary_",
        )

    # read Cherenkov-pool and histogram photon absorbtion-positions
    # -------------------------------------------------------------
    if True:  # not os.path.exists(hist_path):
        run_handle = mdfl.corsika.cpw.Tario(path=pool_path)
        evth, bunches = next(run_handle)
        all_light_field = mdfl.corsika.init_light_field_from_corsika(
            bunches=bunches
        )

        evth[mdfl.corsika.cpw.I_EVTH_RUN_NUMBER] == job["run"]
        evth[mdfl.corsika.cpw.I_EVTH_EVENT_NUMBER] == job["event"]
        prm_dict = mdfl.corsika.cpw._primaries_to_dict(
            primary_bytes=job["primary_bytes"]
        )[0]
        evth[mdfl.corsika.cpw.I_EVTH_PARTICLE_ID] == prm_dict["particle_id"]
        evth[mdfl.corsika.cpw.I_EVTH_TOTAL_ENERGY_GEV] == prm_dict["energy_GeV"]
        evth[mdfl.corsika.cpw.I_EVTH_ZENITH_RAD] == prm_dict["zenith_rad"]
        evth[mdfl.corsika.cpw.I_EVTH_AZIMUTH_RAD] == prm_dict["azimuth_rad"]
        evth[mdfl.corsika.cpw.I_EVTH_STARTING_DEPTH_G_PER_CM2] == prm_dict["depth_g_per_cm2"]
        evth[mdfl.corsika.cpw.I_EVTH_NUM_DIFFERENT_RANDOM_SEQUENCES] == 4
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(1)] == prm_dict["random_seed"][0]["SEED"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(2)] == prm_dict["random_seed"][1]["SEED"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(3)] == prm_dict["random_seed"][2]["SEED"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(4)] == prm_dict["random_seed"][3]["SEED"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(1)] == prm_dict["random_seed"][0]["CALLS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(2)] == prm_dict["random_seed"][1]["CALLS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(3)] == prm_dict["random_seed"][2]["CALLS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(4)] == prm_dict["random_seed"][3]["CALLS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(1)] == prm_dict["random_seed"][0]["BILLIONS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(2)] == prm_dict["random_seed"][1]["BILLIONS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(3)] == prm_dict["random_seed"][2]["BILLIONS"]
        evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(4)] == prm_dict["random_seed"][3]["BILLIONS"]
        evth[mdfl.corsika.cpw.I_EVTH_NUM_OBSERVATION_LEVELS] == 1
        # print("obslev", evth[mdfl.corsika.cpw.I_EVTH_HEIGHT_OBSERVATION_LEVEL(1)]*1e-2, "m")
        evth[mdfl.corsika.cpw.I_EVTH_NUM_REUSES_OF_CHERENKOV_EVENT] == 1

        mask_inlier = mdfl.light_field_characterization.light_field_density_cut(
            light_field=all_light_field, density_cut=job["density_cut"]
        )

        light_field = all_light_field[mask_inlier]
        del(all_light_field)

        stats_on_the_fly = mdfl.light_field_characterization.parameterize_light_field(
            light_field=light_field
        )

        if (
            np.abs(
                stats_on_the_fly["position_med_x_m"]
                - job["shower_statistic"]["position_med_x_m"][0]
            )
            < 1
        ):
            cmap = "inferno"
        else:
            cmap = "Greys"

        hist = np.histogram2d(
            x=light_field["x"],
            y=light_field["y"],
            bins=(job["xy_bin_edges"], job["xy_bin_edges"]),
            weights=light_field["size"],
        )[0]
        hist = hist.astype(np.float32)

        print(job["run"], job["event"], job["skey"], job["ckey"], cmap)
        """
        for statkey in stats_on_the_fly:
            print(
                statkey,
                stats_on_the_fly[statkey],
                job["shower_statistic"][statkey][0],
            )
        """

        ell_maj = job["shower_statistic"]["position_std_major_m"]
        ell_min = job["shower_statistic"]["position_std_minor_m"]
        ell_x = job["shower_statistic"]["position_med_x_m"]
        ell_y = job["shower_statistic"]["position_med_y_m"]
        ell_phi = job["shower_statistic"]["position_phi_rad"]

        ell = plt_patches.Ellipse(
            xy=[ell_x, ell_y],
            width=ell_maj,
            height=ell_min,
            angle=np.rad2deg(ell_phi),
            alpha=0.5,
        )

        fig = sebplt.figure({"rows": 1080, "cols": 1920, "fontsize": 1.5})
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.15, 0.8*(9/16), 0.8))
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.15, 0.02, 0.8])
        ax.plot(
            stats_on_the_fly["position_med_x_m"],
            stats_on_the_fly["position_med_y_m"],
            "xr",
        )
        pcm = ax.pcolormesh(
            job["xy_bin_edges"], job["xy_bin_edges"], hist.T, cmap=cmap,
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        sebplt.plt.colorbar(pcm, cax=ax_cb, extend="max")
        ax.set_xlabel("x" + CFG["plotting"]["label_unit_seperator"] + "m")
        ax.set_ylabel("y" + CFG["plotting"]["label_unit_seperator"] + "m")
        ax.add_artist(ell)
        fig.savefig(hist_path)
        sebplt.close_figure(fig)


for job in jobs:
    run_job(job)
