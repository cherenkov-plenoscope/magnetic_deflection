#!/usr/bin/python
import sys
from os.path import join as opj
import os
import pandas as pd
import numpy as np
import json
import magnetic_deflection as mdfl
import sebastians_matplotlib_addons as sebplt
import plenoirf as irf
import scipy


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot_cherenkov_density")
os.makedirs(out_dir, exist_ok=True)

deflection_table = mdfl.tools.read_deflection_table(os.path.join(work_dir, "raw"))
shower_statistics = mdfl.read_shower_statistics(work_dir=work_dir)

with open(os.path.join(work_dir, "sites.json"), "rt") as f:
    sites = json.loads(f.read())

with open(os.path.join(work_dir, "particles.json"), "rt") as f:
    particles = json.loads(f.read())

with open(os.path.join(work_dir, "pointing.json"), "rt") as f:
    pointing = json.loads(f.read())


deflection_table = mdfl.analysis.cut_invalid_from_deflection_table(
    deflection_table=deflection_table, but_keep_site="Off"
)
deflection_table = mdfl.analysis.add_density_fields_to_deflection_table(
    deflection_table=deflection_table
)

def make_site_str(skey, site):
    return "".join(
        [
            "{:s}, {:.1f}$\,$km$\,$a.s.l., ",
            "Atm.-id {:d}, ",
            "Bx {:.1f}$\,$uT, ",
            "Bz {:.1f}$\,$uT",
        ]
    ).format(
        skey,
        site["observation_level_asl_m"] * 1e-3,
        site["atmosphere_id"],
        site["earth_magnetic_field_x_muT"],
        site["earth_magnetic_field_z_muT"],
    )


figsize = (16 / 2, 9 / 2)
dpi = 240
ax_size = (0.15, 0.12, 0.8, 0.8)

figsize_fit = (9 / 2, 9 / 2)
dpi = 240
ax_size_fit = (0.2, 0.12, 0.75, 0.8)


for skey in deflection_table:
    for pkey in deflection_table[skey]:
        print(skey, pkey)
        site_str = make_site_str(skey, sites[skey])

        azimuths_deg_steps = np.linspace(0, 360, 12, endpoint=False)
        if pkey == "gamma":
            fov_deg = 1.0
        elif pkey == "proton":
            fov_deg = 10.0
        elif pkey == "helium":
            fov_deg = 10.0
        else:
            fov_deg = 90.0
        fov = np.deg2rad(fov_deg)
        rfov = np.sin(fov)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes((0.07, 0.07, 0.85, 0.85))
        cmap_ax = fig.add_axes((0.8, 0.07, 0.02, 0.85))
        if PLOT_TITLE_INFO:
            ax.set_title(site_str, alpha=0.5)
        sebplt.hemisphere.ax_add_grid(
            ax=ax,
            azimuths_deg=azimuths_deg_steps,
            zeniths_deg=np.linspace(0, fov_deg, 10),
            linewidth=1.4,
            color="k",
            alpha=0.1,
            draw_lower_horizontal_edge_deg=fov_deg,
        )
        cmap_name = "nipy_spectral"
        cmap_norm = plt_colors.LogNorm(
            vmin=np.min(t["particle_energy_GeV"]),
            vmax=np.max(t["particle_energy_GeV"]),
        )
        cmap_mappable = matplotlib.cm.ScalarMappable(
            norm=cmap_norm, cmap=cmap_name
        )
        plt.colorbar(cmap_mappable, cax=cmap_ax)
        cmap_ax.set_xlabel("energy$\,/\,$GeV")
        rgbas = cmap_mappable.to_rgba(t["particle_energy_GeV"])
        rgbas[:, 3] = 0.25
        sebplt.hemisphere.ax_add_points(
            ax=ax,
            azimuths_deg=t["particle_azimuth_deg"],
            zeniths_deg=t["particle_zenith_deg"],
            point_diameter=0.1 * rfov,
            rgbas=rgbas,
        )
        if PLOT_TITLE_INFO_SKY_DOME:
            ax.text(
                -1.6 * rfov,
                0.65 * rfov,
                "direction of particle\n\nazimuth w.r.t.\nmagnetic north",
            )
        ax.text(
            -1.0 * rfov, -1.0 * rfov, "zenith {:1.0f}$^\circ$".format(fov_deg)
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
        sebplt.hemisphere.ax_add_ticklabels(
            ax=ax, azimuths_deg=azimuths_deg_steps, rfov=rfov
        )
        ax.set_xlim([-1.01 * rfov, 1.01 * rfov])
        ax.set_ylim([-1.01 * rfov, 1.01 * rfov])
        fig.savefig(
            os.path.join(
                out_dir,
                "{:s}_{:s}_{:s}.jpg".format(skey, pkey, "dome"),
            )
        )
        plt.close(fig)

    density_map = {
        "num_cherenkov_photons_per_shower": {
            "label": "size of Cherenkov-pool$\,/\,$1"
        },
        "spread_area_m2": {
            "label": "Cherenkov-pool's spread in area$\,/\,$m$^{2}$"
        },
        "spread_solid_angle_deg2": {
            "label": "Cherenkov-pool's spread in solid angle$\,/\,$deg$^{2}$"
        },
        "light_field_outer_density": {
            "label": "density of Cherenkov-pool$\,/\,$m$^{-2}\,$deg$^{-2}$"
        },
    }

    parmap = {"gamma": "k", "electron": "b", "proton": "r", "helium": "orange"}

    for den_key in density_map:
        ts = deflection_table[skey]
        alpha = 0.2
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(ax_size)
        for pkey in parmap:
            ax.plot(
                ts[pkey]["particle_energy_GeV"],
                ts[pkey][den_key],
                "o",
                color=parmap[pkey],
                alpha=alpha,
                label=pkey,
            )
        leg = ax.legend()
        if PLOT_TITLE_INFO:
            ax.set_title(site_str, alpha=0.5)
        ax.loglog()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("energy$\,/\,$GeV")
        ax.set_xlim([0.4, 200.0])
        ax.set_ylim([1e-6, 1e3])
        ax.set_ylabel(density_map[den_key]["label"])
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(work_dir, "{:s}_{:s}.jpg".format(skey, den_key))
        )
        plt.close(fig)


"""
density by side
"""
den_key = "light_field_outer_density"

sitemap = {"namibia": "+", "lapalma": "^", "chile": "*", "namibiaOff": "."}


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


particle_colors = {
    "electron": "blue",
    "gamma": "black",
}

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes(ax_size)
for pkey in ["electron", "gamma"]:
    for skey in nice_site_labels:
        E = deflection_table[skey][pkey]["particle_energy_GeV"]
        V = deflection_table[skey][pkey][den_key]
        mask = np.arange(20, len(E), len(E) // 10)
        ax.plot(
            E,
            V,
            sitemap[skey],
            color=particle_colors[pkey],
            alpha=0.1 * alpha,
        )
        if particle_colors[pkey] == "black":
            label = nice_site_labels[skey]
        else:
            label = None
        ax.plot(
            E[mask],
            smooth(V, 9)[mask],
            sitemap[skey],
            color=particle_colors[pkey],
            label=label,
        )

ax.text(1.1, 50, "gamma-ray", color=particle_colors["gamma"])
ax.text(1.1, 25, "electron", color=particle_colors["electron"])
leg = ax.legend()
ax.loglog()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("energy$\,/\,$GeV")
ax.set_xlim([0.4, 20.0])
ax.set_ylim([1e-3, 1e2])
ax.set_ylabel(density_map[den_key]["label"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join(out_dir, "{:s}.jpg".format(den_key)))
plt.close(fig)
