#!/usr/bin/python
import sys
import os
import numpy as np
import pandas as pd
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
figsize = {"rows": 720, "cols": 1280, "fontsize": 1.25}

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot_cherenkov_density")
os.makedirs(out_dir, exist_ok=True)

shower_statistics = mdfl.read_shower_statistics(work_dir=work_dir)
c = mdfl.read_config(work_dir=work_dir)

MIN_NUM_SHOWER = 11

density_map = {
    "num_photons": {"label": "intensity$\,/\,$1", "lim": [1e2, 1e6],},
    "spread_area_m2": {"label": "area$\,/\,$m$^{2}$", "lim": [1e4, 1e7],},
    "spread_solid_angle_deg2": {
        "label": "solid angle$\,/\,(1^\circ)^{2}$",
        "lim": [1e-1, 1e2],
    },
    "light_field_outer_density": {
        "label": "density$\,/\,$m$^{-2}\,(1^\circ)^{-2}$",
        "lim": [1e-3, 1e2],
    },
}

pc = {
    "sites": {
        "namibia": {
            "nice_label": "Gamsberg",
            "marker": "+",
            "linestyle": "--",
        },
        "chile": {
            "nice_label": "Chajnantor",
            "marker": "*",
            "linestyle": ":",
        },
        "namibiaOff": {
            "nice_label": "Gamsberg-Off",
            "marker": ".",
            "linestyle": "-.",
        },
        "lapalma": {
            "nice_label": "Roque",
            "marker": "^",
            "linestyle": "-",
        },
    },
    "particles": {
        "gamma": {
            "color": "black",
            "nice_label": "gamma-ray",
        },
        "electron": {
            "color": "blue",
            "nice_label": "electron",
        },
        "proton": {
            "color": "red",
            "nice_label": "proton",
        },
        "helium": {
            "color": "orange",
            "nice_label": "helium",
        },
    }
}

ENERGY = {}
ENERGY["fine"] = {}
ENERGY["fine"]["num_bins"] = 60
ENERGY["fine"]["bin_edges"] = np.geomspace(
    1e-1, 1e2, ENERGY["fine"]["num_bins"] + 1
)
ENERGY["coarse"] = {}
ENERGY["coarse"]["num_bins"] = 10
ENERGY["coarse"]["bin_edges"] = np.geomspace(
    1e-1, 1e2, ENERGY["coarse"]["num_bins"] + 1
)

for skey in shower_statistics:
    for pkey in shower_statistics[skey]:
        statkeys = list(shower_statistics[skey][pkey].dtype.names)

def make_off_axis_angle_deg(
    cherenkov_direction_med_cx_rad,
    cherenkov_direction_med_cy_rad,
    pointing_azimuth_deg,
    pointing_zenith_deg,
):
    (
        cer_azimuth_deg,
        cer_zenith_deg,
    ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
        cx=cherenkov_direction_med_cx_rad, cy=cherenkov_direction_med_cy_rad,
    )

    return mdfl.spherical_coordinates._angle_between_az_zd_deg(
        az1_deg=cer_azimuth_deg,
        zd1_deg=cer_zenith_deg,
        az2_deg=pointing_azimuth_deg,
        zd2_deg=pointing_zenith_deg,
    )

# cut on-axis
# -----------
on_axis_shower_statistics = {}
for skey in shower_statistics:
    on_axis_shower_statistics[skey] = {}
    for pkey in shower_statistics[skey]:

        sst = shower_statistics[skey][pkey]

        off_axis_deg = make_off_axis_angle_deg(
            cherenkov_direction_med_cx_rad=sst["direction_med_cx_rad"],
            cherenkov_direction_med_cy_rad=sst["direction_med_cy_rad"],
            pointing_azimuth_deg=c["pointing"]["azimuth_deg"],
            pointing_zenith_deg=c["pointing"]["zenith_deg"],
        )

        on_axis_mask = (
            off_axis_deg
            <= 2
            * c["particles"][pkey]["magnetic_deflection_max_off_axis_deg"]
        )
        on_axis_shower_statistics[skey][pkey] = shower_statistics[skey][pkey][on_axis_mask]

del(shower_statistics)


for tkey in statkeys:
    for skey in on_axis_shower_statistics:

        fig = sebplt.figure(figsize)
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

        for pkey in on_axis_shower_statistics[skey]:
            oasst = on_axis_shower_statistics[skey][pkey]
            ax.plot(
                oasst["particle_energy_GeV"],
                oasst[tkey],
                marker=".",
                markersize=0.1,
                linewidth=0.0,
                color=pc["particles"][pkey]["color"],
                alpha=0.1,
            )

        ax.semilogx()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("energy$\,/\,$GeV")
        ax.set_ylabel(tkey)
        ax.set_xlim(
            [
                min(ENERGY["fine"]["bin_edges"]),
                max(ENERGY["fine"]["bin_edges"]),
            ]
        )
        #ax.set_ylim(1e2, 1e6)
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(skey, tkey)))
        plt.close(fig)




# compute density
# ---------------
cherenkov_density = {}
for skey in on_axis_shower_statistics:
    cherenkov_density[skey] = {}
    for pkey in on_axis_shower_statistics[skey]:
        oasst = on_axis_shower_statistics[skey][pkey]

        den = {}
        den["num_photons"] = oasst["num_photons"]
        den["spread_area_m2"] = (
            np.pi * oasst["position_std_major_m"] * oasst["position_std_minor_m"]
        )
        den["spread_solid_angle_deg2"] = (
            np.pi
            * np.rad2deg(oasst["direction_std_major_rad"])
            * np.rad2deg(oasst["direction_std_minor_rad"])
        )
        den["light_field_outer_density"] = oasst["num_photons"] / (
            den["spread_solid_angle_deg2"] * den["spread_area_m2"]
        )

        cherenkov_density[skey][pkey] = pd.DataFrame(den).to_records(
            index=False
        )



# cut on-axis and bin in energy
# -----------------------------
ooo = {}
for fkey in ENERGY:

    num_energy_bins = 60
    energy_bin_edges = np.geomspace(1e-1, 1e2, num_energy_bins + 1)

    ooo[fkey] = {}
    for skey in cherenkov_density:
        ooo[fkey][skey] = {}
        for pkey in cherenkov_density[skey]:
            ooo[fkey][skey][pkey] = {}
            oasst = on_axis_shower_statistics[skey][pkey]

            for dkey in density_map:
                ooo[fkey][skey][pkey][dkey] = {
                    "percentile84": np.nan * np.ones(ENERGY[fkey]["num_bins"]),
                    "percentile50": np.nan * np.ones(ENERGY[fkey]["num_bins"]),
                    "percentile16": np.nan * np.ones(ENERGY[fkey]["num_bins"]),
                    "num": np.zeros(ENERGY[fkey]["num_bins"]),
                }

            for ebin in range(ENERGY[fkey]["num_bins"]):
                E_start = ENERGY[fkey]["bin_edges"][ebin]
                E_stop = ENERGY[fkey]["bin_edges"][ebin + 1]
                E_mask = np.logical_and(
                    oasst["particle_energy_GeV"] >= E_start,
                    oasst["particle_energy_GeV"] < E_stop,
                )
                num_samples = np.sum(E_mask)

                print(skey, pkey, "E-bin:", ebin, "num. shower:", num_samples)

                for dkey in density_map:
                    cde = cherenkov_density[skey][pkey][dkey]
                    valid_cde = cde[E_mask]

                    if num_samples >= MIN_NUM_SHOWER:
                        p84 = np.percentile(valid_cde, 50 + 68 / 2)
                        p50 = np.percentile(valid_cde, 50)
                        p16 = np.percentile(valid_cde, 50 - 68 / 2)
                        ooo[fkey][skey][pkey][dkey]["percentile84"][ebin] = p84
                        ooo[fkey][skey][pkey][dkey]["percentile50"][ebin] = p50
                        ooo[fkey][skey][pkey][dkey]["percentile16"][ebin] = p16
                        ooo[fkey][skey][pkey][dkey]["num"][ebin] = num_samples

oof = ooo["fine"]
ooc = ooo["coarse"]

alpha = 0.2

for dkey in density_map:
    for skey in cherenkov_density:

        fig = sebplt.figure(figsize)
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

        for pkey in cherenkov_density[skey]:

            for ebin in range(num_energy_bins):
                E_start = ENERGY["fine"]["bin_edges"][ebin]
                E_stop = ENERGY["fine"]["bin_edges"][ebin + 1]

                if oof[skey][pkey][dkey]["num"][ebin] > MIN_NUM_SHOWER:
                    p16 = oof[skey][pkey][dkey]["percentile16"][ebin]
                    p50 = oof[skey][pkey][dkey]["percentile50"][ebin]
                    p84 = oof[skey][pkey][dkey]["percentile84"][ebin]

                    ax.fill(
                        [E_start, E_start, E_stop, E_stop],
                        [p16, p84, p84, p16],
                        color=pc["particles"][pkey]["color"],
                        alpha=alpha,
                        linewidth=0.0,
                    )
                    ax.plot(
                        [E_start, E_stop],
                        [p50, p50],
                        color=pc["particles"][pkey]["color"],
                        alpha=1.0,
                    )
        ax.loglog()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("energy$\,/\,$GeV")
        ax.set_ylabel(density_map[dkey]["label"])
        ax.set_xlim(
            [
                min(ENERGY["fine"]["bin_edges"]),
                max(ENERGY["fine"]["bin_edges"]),
            ]
        )
        ax.set_ylim(density_map[dkey]["lim"])
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(skey, dkey)))
        plt.close(fig)


# statistics
# ----------
for skey in oof:

    fig = sebplt.figure(figsize)
    ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

    for pkey in oof[skey]:
        for ebin in range(num_energy_bins):
            E_start = ENERGY["fine"]["bin_edges"][ebin]
            E_stop = ENERGY["fine"]["bin_edges"][ebin + 1]

            count = oof[skey][pkey][dkey]["num"][ebin]
            ax.plot(
                [E_start, E_stop],
                [count, count],
                color=pc["particles"][pkey]["color"],
                alpha=1.0,
            )
            ax.fill(
                [E_start, E_start, E_stop, E_stop],
                [0, count, count, 0],
                color=pc["particles"][pkey]["color"],
                alpha=alpha,
                linewidth=0.0,
            )

    ax.loglog()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("energy$\,/\,$GeV")
    ax.set_ylabel("num. shower / 1")
    ax.set_xlim(
        [min(ENERGY["fine"]["bin_edges"]), max(ENERGY["fine"]["bin_edges"])]
    )
    ax.set_ylim([1e1, 1e5])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(os.path.join(out_dir, "{:s}_statistics.jpg".format(skey)))
    plt.close(fig)


# density by side
# ---------------

dkey = "light_field_outer_density"

fig = sebplt.figure(figsize)
ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

for skey in ooc:
    for pkey in ooc[skey]:

        if pc["particles"][pkey]["color"] == "black":
            site_label = pc["sites"][skey]["nice_label"]
        else:
            site_label = None

        ax.plot(
            ENERGY["coarse"]["bin_edges"][0:-1],
            ooc[skey][pkey][dkey]["percentile50"],
            marker=pc["sites"][skey]["marker"],
            linewidth=0.0,
            color=pc["particles"][pkey]["color"],
            alpha=0.5,
            label=site_label,
        )

        ax.plot(
            ENERGY["coarse"]["bin_edges"][0:-1],
            ooc[skey][pkey][dkey]["percentile50"],
            marker=None,
            linestyle=pc["sites"][skey]["linestyle"],
            linewidth=0.5,
            color=pc["particles"][pkey]["color"],
            alpha=0.33,
        )

leg = ax.legend()
ax.loglog()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("energy$\,/\,$GeV")
ax.set_xlim(
    [min(ENERGY["coarse"]["bin_edges"]), max(ENERGY["coarse"]["bin_edges"])]
)
ax.set_ylim(density_map[dkey]["lim"])
ax.set_ylabel(density_map[dkey]["label"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join(out_dir, "{:s}_all_sites.jpg".format(dkey)))
plt.close(fig)
