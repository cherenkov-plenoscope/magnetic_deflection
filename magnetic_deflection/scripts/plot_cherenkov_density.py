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

parmap = {"gamma": "k", "electron": "b", "proton": "r", "helium": "orange"}

MIN_NUM_SHOWER = 11

# compute density
# ---------------
cherenkov_density = {}
for skey in shower_statistics:
    cherenkov_density[skey] = {}
    for pkey in shower_statistics[skey]:
        sst = shower_statistics[skey][pkey]

        den = {}
        den["num_photons"] = sst["num_photons"]
        den["spread_area_m2"] = (
            np.pi * sst["position_std_major_m"] * sst["position_std_minor_m"]
        )
        den["spread_solid_angle_deg2"] = (
            np.pi
            * np.rad2deg(sst["direction_std_major_rad"])
            * np.rad2deg(sst["direction_std_minor_rad"])
        )
        den["light_field_outer_density"] = sst["num_photons"] / (
            den["spread_solid_angle_deg2"] * den["spread_area_m2"]
        )

        cherenkov_density[skey][pkey] = pd.DataFrame(den).to_records(
            index=False
        )


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

ENERGY = {}
ENERGY["fine"] = {}
ENERGY["fine"]["num_bins"] = 60
ENERGY["fine"]["bin_edges"] = np.geomspace(1e-1, 1e2, ENERGY["fine"]["num_bins"] + 1)
ENERGY["coarse"] = {}
ENERGY["coarse"]["num_bins"] = 10
ENERGY["coarse"]["bin_edges"] = np.geomspace(1e-1, 1e2, ENERGY["coarse"]["num_bins"] + 1)

num_energy_bins = 60
energy_bin_edges = np.geomspace(1e-1, 1e2, num_energy_bins + 1)

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
                    sst["particle_energy_GeV"] >= E_start,
                    sst["particle_energy_GeV"] < E_stop,
                )
                mask = np.logical_and(E_mask, on_axis_mask)
                num_samples = np.sum(mask)

                print(skey, pkey, "E-bin:", ebin, "num. shower:", num_samples)

                for dkey in density_map:
                    cde = cherenkov_density[skey][pkey][dkey]
                    valid_cde = cde[mask]

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
    for skey in shower_statistics:

        fig = sebplt.figure(figsize)
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

        for pkey in parmap:

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
                        color=parmap[pkey],
                        alpha=alpha,
                        linewidth=0.0,
                    )
                    ax.plot(
                        [E_start, E_stop],
                        [p50, p50],
                        color=parmap[pkey],
                        alpha=1.0,
                    )
        ax.loglog()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("energy$\,/\,$GeV")
        ax.set_ylabel(density_map[dkey]["label"])
        ax.set_xlim([min(ENERGY["fine"]["bin_edges"]), max(ENERGY["fine"]["bin_edges"])])
        ax.set_ylim(density_map[dkey]["lim"])
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(skey, dkey)))
        plt.close(fig)


# statistics
# ----------
for skey in shower_statistics:

    fig = sebplt.figure(figsize)
    ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

    for pkey in parmap:
        for ebin in range(num_energy_bins):
            E_start = ENERGY["fine"]["bin_edges"][ebin]
            E_stop = ENERGY["fine"]["bin_edges"][ebin + 1]

            count = oof[skey][pkey][dkey]["num"][ebin]
            ax.plot(
                [E_start, E_stop],
                [count, count],
                color=parmap[pkey],
                alpha=1.0,
            )
            ax.fill(
                [E_start, E_start, E_stop, E_stop],
                [0, count, count, 0],
                color=parmap[pkey],
                alpha=alpha,
                linewidth=0.0,
            )

    ax.loglog()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("energy$\,/\,$GeV")
    ax.set_ylabel("num. shower / 1")
    ax.set_xlim([min(ENERGY["fine"]["bin_edges"]), max(ENERGY["fine"]["bin_edges"])])
    ax.set_ylim([1e1, 1e5])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(os.path.join(out_dir, "{:s}_statistics.jpg".format(skey)))
    plt.close(fig)



# density by side
# ---------------

nice_site_labels = {
    #"namibiaOff": "Gamsberg-Off",
    "namibia": "Gamsberg",
    "chile": "Chajnantor",
    #"lapalma": "Roque",
}

dkey = "light_field_outer_density"

sitemap = {"namibia": "+", "lapalma": "^", "chile": "*", "namibiaOff": "."}

particle_colors = {
    "electron": "blue",
    "gamma": "black",
}

fig = sebplt.figure(figsize)
ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

for skey in nice_site_labels:
    for pkey in particle_colors:

        if particle_colors[pkey] == "black":
            label = nice_site_labels[skey]
        else:
            label = None

        ax.plot(
            ENERGY["coarse"]["bin_edges"][0:-1],
            ooc[skey][pkey][dkey]["percentile50"],
            sitemap[skey],
            color=particle_colors[pkey],
            alpha=0.5,
            label=label,
        )

        ax.plot(
            ENERGY["coarse"]["bin_edges"][0:-1],
            ooc[skey][pkey][dkey]["percentile50"],
            "-",
            color=particle_colors[pkey],
            alpha=0.33,
        )

leg = ax.legend()
ax.loglog()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("energy$\,/\,$GeV")
ax.set_xlim([min(ENERGY["coarse"]["bin_edges"]), max(ENERGY["coarse"]["bin_edges"])])
ax.set_ylim(density_map[dkey]["lim"])
ax.set_ylabel(density_map[dkey]["label"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join(out_dir, "{:s}_all_sites.jpg".format(dkey)))
plt.close(fig)
