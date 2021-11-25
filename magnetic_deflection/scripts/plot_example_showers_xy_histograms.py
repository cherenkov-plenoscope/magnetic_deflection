#!/usr/bin/python
import sys
import os
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib
import pickle
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

pkey = "helium"
dkey = "cherenkov_area_m2"

cases = {
    "LowEnergySmallArea": {
        "energy": {"start": 8, "stop": 16},
        dkey: {"start": 1e6, "stop": 5e6},
    },
    "HighEnergyLargeArea": {
        "energy": {"start": 32, "stop": 64},
        dkey: {"start": 1e7, "stop": 1e8},
    },
}

XY_BIN_EDGES = np.linspace(-1e4, 1e4, 401)
ON_AXIS_SCALE = 1.0
MAX_NUM_EXAMPLES = 8

def margin(a, b, epsilon):
    match = np.abs(a - b) < epsilon
    if not match:
        print("a", a, "b", b, "abs(a-b)", np.abs(a - b), "epsilon", epsilon)
    return match

# find example events
# -------------------
shower_statistics = mdfl.read_statistics(work_dir=work_dir)
shower_explicit_steerings = mdfl.read_explicit_steerings(work_dir=work_dir)


example_events_path = os.path.join(out_dir, "example_events.pkl")
if not os.path.exists(example_events_path):
    print("Find example_events")
    example_events = {}
    for skey in CFG["sites"]:
        example_events[skey] = {}
        for ckey in cases:
            example_events[skey][ckey] = {}
            sp_stats = shower_statistics[skey][pkey]
            E_cut = cases[ckey]["energy"]
            A_cut = cases[ckey][dkey]

            for i in range(len(sp_stats)):
                E = sp_stats["particle_energy_GeV"][i]
                A = np.pi * sp_stats["cherenkov_radius50_m"][i] ** 2
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
    with open(example_events_path, "wb") as f:
        f.write(pickle.dumps(example_events))
else:
    print("Load existing example_events")
    with open(example_events_path, "rb") as f:
        example_events = pickle.loads(f.read())


print("Found example candidates:")
for skey in CFG["sites"]:
    for ckey in cases:
        print(
            "{:>20s} {:>20s}, num: {: 6d}".format(
                skey, ckey, len(example_events[skey][ckey])
            )
        )


print("Limit number example events:")
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


print("Get steering for example events:")
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


print("Reproduce example events:")
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
    job_key = "{:06d}_{:06d}".format(
        job["run"],
        job["event"],
    )

    pool_path = os.path.join(job_dir, job_key + "_cherenkov_pool.tar")
    hist_path = os.path.join(
        out_dir,
        "{:s}_{:s}_{:s}".format(
            job["skey"],
            job["pkey"],
            job["ckey"],
        ) + "_" + job_key + "_cherenkov_pool_histogram_xy.jpg"
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
        light_field = mdfl.corsika.init_light_field_from_corsika(
            bunches=bunches
        )

        assert evth[mdfl.corsika.cpw.I_EVTH_RUN_NUMBER] == job["run"]
        assert evth[mdfl.corsika.cpw.I_EVTH_EVENT_NUMBER] == job["event"]
        prm_dict = mdfl.corsika.cpw._primaries_to_dict(
            primary_bytes=job["primary_bytes"]
        )[0]
        assert evth[mdfl.corsika.cpw.I_EVTH_PARTICLE_ID] == prm_dict["particle_id"]
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_TOTAL_ENERGY_GEV], prm_dict["energy_GeV"], 1e-3)
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_ZENITH_RAD], prm_dict["zenith_rad"], 1e-3)

        # assert margin(evth[mdfl.corsika.cpw.I_EVTH_AZIMUTH_RAD], prm_dict["azimuth_rad"], 1e-3)
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_STARTING_DEPTH_G_PER_CM2], prm_dict["depth_g_per_cm2"], 1e-3)
        assert evth[mdfl.corsika.cpw.I_EVTH_NUM_DIFFERENT_RANDOM_SEQUENCES] == 4
        assert prm_dict["random_seed"][0]["SEED"] == job["run"] * 100000 + job["event"]
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(1)], prm_dict["random_seed"][0]["SEED"], 35)
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(2)], prm_dict["random_seed"][1]["SEED"], 35)
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(3)], prm_dict["random_seed"][2]["SEED"], 35)
        assert margin(evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED(4)], prm_dict["random_seed"][3]["SEED"], 35)
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(1)] == prm_dict["random_seed"][0]["CALLS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(2)] == prm_dict["random_seed"][1]["CALLS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(3)] == prm_dict["random_seed"][2]["CALLS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_CALLS(4)] == prm_dict["random_seed"][3]["CALLS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(1)] == prm_dict["random_seed"][0]["BILLIONS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(2)] == prm_dict["random_seed"][1]["BILLIONS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(3)] == prm_dict["random_seed"][2]["BILLIONS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_RANDOM_SEED_BILLIONS(4)] == prm_dict["random_seed"][3]["BILLIONS"]
        assert evth[mdfl.corsika.cpw.I_EVTH_NUM_OBSERVATION_LEVELS] == 1
        # print("obslev", evth[mdfl.corsika.cpw.I_EVTH_HEIGHT_OBSERVATION_LEVEL(1)]*1e-2, "m")
        assert evth[mdfl.corsika.cpw.I_EVTH_NUM_REUSES_OF_CHERENKOV_EVENT] == 1

        if len(light_field["x"]) < 100:
            reproduction_valid = False
            return 0

        stats_on_the_fly = mdfl.light_field_characterization.parameterize_light_field(
            light_field=light_field
        )

        if (
            np.abs(
                stats_on_the_fly["cherenkov_x_m"]
                - job["shower_statistic"]["cherenkov_x_m"][0]
            )
            < 1e0
            and
            np.abs(
                stats_on_the_fly["cherenkov_y_m"]
                - job["shower_statistic"]["cherenkov_y_m"][0]
            )
            < 1e0
        ):
            reproduction_valid = True
        else:
            reproduction_valid = False

        _seed_int64 = np.int64(
            np.int64(job["run"]) * 100 * 1000 + np.int64(job["event"])
        )
        _seed_int32 = np.int32(
            np.int32(job["run"]) * 100 * 1000 + np.int32(job["event"])
        )
        _seed_float32 = np.float32(_seed_int64)
        _seed_diff = _seed_int64 - _seed_float32

        print(
            "run {: 6d}, event {: 6d}, site {:>16s}, case {:>16s}, valid {:d}, D {:f}".format(
                job["run"], job["event"], job["skey"], job["ckey"], reproduction_valid, _seed_diff
            )
        )

        if not reproduction_valid:
            return 0

        hist = np.histogram2d(
            x=light_field["x"] - job["shower_statistic"]["cherenkov_x_m"][0],
            y=light_field["y"] - job["shower_statistic"]["cherenkov_y_m"][0],
            bins=(job["xy_bin_edges"], job["xy_bin_edges"]),
            weights=light_field["size"],
        )[0]
        hist = hist.astype(np.float32)


        dkey = "cherenkov_area_m2"

        fig = sebplt.figure({"rows": 1080, "cols": 2560, "fontsize": 1.5})
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.15, 0.8*(1080/2560), 0.8))
        ax_cb = sebplt.add_axes(fig=fig, span=[0.5, 0.15, 0.015, 0.8])
        ax_cut = sebplt.add_axes(fig=fig, span=[0.65, 0.15, 0.3, 0.8])

        ax_cut.set_xlabel(
            "energy"
            + CFG["plotting"]["label_unit_seperator"]
            + "GeV"
        )
        ax_cut.set_ylabel(
            CFG["plotting"]["light_field"][dkey]["label"]
            + CFG["plotting"]["label_unit_seperator"]
            + CFG["plotting"]["light_field"][dkey]["unit"]
        )
        ax_cut.set_xlim(1e-1, 1e2)
        ax_cut.set_ylim(CFG["plotting"]["light_field"][dkey]["limits"])
        ax_cut.loglog()
        sebplt.ax_add_box(
            ax=ax_cut,
            xlim=[cases[job["ckey"]]["energy"]["start"], cases[job["ckey"]]["energy"]["stop"]],
            ylim=[cases[job["ckey"]][dkey]["start"], cases[job["ckey"]][dkey]["stop"]],
            color="k",
            linewidth=None
        )
        cherenkov_area_m2 = np.pi * job["shower_statistic"]["cherenkov_radius50_m"][0] ** 2
        ax_cut.plot(
            prm_dict["energy_GeV"],
            cherenkov_area_m2,
            "xk",
        )

        if reproduction_valid:
            cmap = "inferno"
        else:
            cmap = "Greys"
            ax.plot(
                0,
                0,
                "xr",
                markersize=4,
            )
        sebplt.ax_add_circle(
            ax=ax,
            x=0,
            y=0,
            r=job["shower_statistic"]["cherenkov_radius50_m"],
            linewidth=0.5,
            linestyle="-",
            color="green",
            alpha=1,
            num_steps=128,
        )
        pcm = ax.pcolormesh(
            job["xy_bin_edges"], job["xy_bin_edges"], hist.T, cmap=cmap,
            norm=sebplt.plt_colors.PowerNorm(gamma=0.333, vmin=0.0, vmax=1e3),
        )
        sebplt.plt.colorbar(pcm, cax=ax_cb, extend="max")
        ax.set_xlabel("rel. x" + CFG["plotting"]["label_unit_seperator"] + "m")
        ax.set_ylabel("rel. y" + CFG["plotting"]["label_unit_seperator"] + "m")
        fig.savefig(hist_path)
        sebplt.close_figure(fig)
        return 1


for job in jobs:
    run_job(job)
