#!/usr/bin/python
import sys
from os.path import join as opj
import os
import pandas as pd
import numpy as np
import json
import magnetic_deflection as mdfl
from magnetic_deflection import plot_sky_dome as mdfl_plot
import plenoirf as irf
import scipy


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
deflection_dir = argv[1]

deflection_table = mdfl.tools.read_deflection_table(path=deflection_dir)

with open(os.path.join(deflection_dir, "sites.json"), "rt") as f:
    sites = json.loads(f.read())

with open(os.path.join(deflection_dir, "particles.json"), "rt") as f:
    particles = json.loads(f.read())

with open(os.path.join(deflection_dir, "pointing.json"), "rt") as f:
    pointing = json.loads(f.read())

PLOT_TITLE_INFO = False
PLOT_TITLE_INFO_SKY_DOME = False

PLOT_POWER_LAW_FIT = True
POWER_LAW_FIT_COLOR = "k"
PLOT_ENERGY_SUPPORTS = False
PLOT_RAW_ESTIMATES = True

key_map = {
    "particle_azimuth_deg": {
        "unit": "deg",
        "name": "particle-azimuth",
        "factor": 1,
        "start": 90.0,
        "etend_high_energies": True,
    },
    "particle_zenith_deg": {
        "unit": "deg",
        "name": "particle-zenith",
        "factor": 1,
        "start": 0.0,
        "etend_high_energies": True,
    },
    "cherenkov_pool_x_m": {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
        "start": 0.0,
        "etend_high_energies": True,
    },
    "cherenkov_pool_y_m": {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
        "start": 0.0,
        "etend_high_energies": True,
    },
}

nice_site_labels = {
    "namibiaOff": "Gamsberg-Off",
    "namibia": "Gamsberg",
    "chile": "Chajnantor",
    "lapalma": "Roque",
}

nice_variable_keys = {
    "particle_azimuth_deg": "$\\PrmAz{}$\\,/\\,deg",
    "particle_zenith_deg": "$\\PrmZd{}$\\,/\\,deg",
    "cherenkov_pool_x_m": "$\\CerX{}$\\,/\\,m",
    "cherenkov_pool_y_m": "$\\CerY{}$\\,/\\,m",
}

nice_pkeys = {
    "gamma": "Gamma-ray",
    "electron": "Electron",
    "proton": "Proton",
    "helium": "Helium",
}

nice_power_law_variable_keys = {
    "A": "$\\PowerLawA{}$",
    "B": "$\\PowerLawB{}$",
    "C": "$\\PowerLawC{}$",
}


charge_signs = {}
for pkey in particles:
    charge_signs[pkey] = np.sign(particles[pkey]["electric_charge_qe"])


figsize = (16 / 2, 9 / 2)
dpi = 240
ax_size = (0.15, 0.12, 0.8, 0.8)

figsize_fit = (9 / 2, 9 / 2)
dpi = 240
ax_size_fit = (0.2, 0.12, 0.75, 0.8)


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


power_law_fit_table = {}


def latex_table(matrix):
    out = ""
    for line in matrix:
        for idx, item in enumerate(line):
            out += item
            if idx + 1 == len(line):
                out += " \\\\"
            else:
                out += " & "
        out += "\n"
    return out


def latex_format_scientific(f):
    pass


def make_latex_table_with_power_law_fit(power_law_fit_table):
    matrix = []
    partile_line = ["", "", ""]
    for pkey in nice_pkeys:
        partile_line.append(nice_pkeys[pkey])
    matrix.append(partile_line)
    for skey in nice_site_labels:
        if "Off" in skey:
            continue
        site_line = [nice_site_labels[skey], "", "", "", "", "", ""]
        matrix.append(site_line)
        for variable_key in nice_variable_keys:
            variable_line = [
                "",
                nice_variable_keys[variable_key],
                "",
                "",
                "",
                "",
                "",
            ]
            matrix.append(variable_line)

            for param_key in ["A", "B", "C"]:
                value_line = ["", "", nice_power_law_variable_keys[param_key]]
                for pkey in nice_pkeys:

                    power_law_fit = power_law_fit_table[skey][pkey][
                        variable_key
                    ]["power_law"]

                    if param_key == "B":
                        value = "{:,.3f}".format(power_law_fit[param_key])
                    else:
                        value = "{:,.1f}".format(power_law_fit[param_key])
                    value_line.append(value)
                matrix.append(value_line)
    return latex_table(matrix)


deflection_table = mdfl.analysis.cut_invalid_from_deflection_table(
    deflection_table=deflection_table, but_keep_site="Off"
)
deflection_table = mdfl.analysis.add_density_fields_to_deflection_table(
    deflection_table=deflection_table
)


for skey in deflection_table:
    power_law_fit_table[skey] = {}
    for pkey in deflection_table[skey]:
        print(skey, pkey)
        site_str = make_site_str(skey, sites[skey])

        t = deflection_table[skey][pkey]
        energy_fine = np.geomspace(
            np.min(t["particle_energy_GeV"]),
            10 * np.max(t["particle_energy_GeV"]),
            1000,
        )

        if "Off" in skey:
            continue

        power_law_fit_table[skey][pkey] = {}
        for key in key_map:
            sres = mdfl.analysis.smooth(
                energies=t["particle_energy_GeV"], values=t[key]
            )
            energy_supports = sres["energy_supports"]
            key_std80 = sres["key_std80"]
            key_mean80 = sres["key_mean80"]
            unc80_upper = key_mean80 + key_std80
            unc80_lower = key_mean80 - key_std80

            if pkey == "electron":
                valid_range = energy_supports > 1.0
                energy_supports = energy_supports[valid_range]
                key_std80 = key_std80[valid_range]
                key_mean80 = key_mean80[valid_range]
                unc80_upper = unc80_upper[valid_range]
                unc80_lower = unc80_lower[valid_range]

            key_start = charge_signs[pkey] * key_map[key]["start"]

            if key_map[key]["etend_high_energies"]:
                energy_bins_ext = np.array(
                    energy_supports.tolist()
                    + np.geomspace(200, 600, 20).tolist()
                )
                key_mean80_ext = np.array(
                    key_mean80.tolist() + (key_start * np.ones(20)).tolist()
                )
            else:
                energy_bins_ext = energy_supports.copy()
                key_mean80_ext = key_mean80.copy()

            if np.mean(key_mean80 - key_start) > 0:
                sig = -1
            else:
                sig = 1

            expy, pcov = scipy.optimize.curve_fit(
                mdfl.analysis.power_law,
                energy_bins_ext,
                key_mean80_ext - key_start,
                p0=(sig * charge_signs[pkey], 1.0),
            )

            info_str = pkey + ", " + site_str

            rec_key = mdfl.analysis.power_law(
                energy=energy_fine, scale=expy[0], index=expy[1]
            )
            rec_key += key_start

            fig = plt.figure(figsize=figsize_fit, dpi=dpi)
            ax = fig.add_axes(ax_size_fit)
            if PLOT_RAW_ESTIMATES:
                ax.plot(
                    t["particle_energy_GeV"],
                    np.array(t[key]) * key_map[key]["factor"],
                    "ko",
                    alpha=0.05,
                )

            if PLOT_ENERGY_SUPPORTS:
                ax.plot(
                    energy_supports, key_mean80 * key_map[key]["factor"], "kx"
                )
                for ibin in range(len(energy_supports)):
                    _x = energy_supports[ibin]
                    _y_low = unc80_lower[ibin]
                    _y_high = unc80_upper[ibin]
                    ax.plot(
                        [_x, _x],
                        np.array([_y_low, _y_high]) * key_map[key]["factor"],
                        "k-",
                    )
                ax.plot(
                    energy_bins_ext,
                    key_mean80_ext * key_map[key]["factor"],
                    "bo",
                    alpha=0.3,
                )

            if PLOT_POWER_LAW_FIT:
                ax.plot(
                    energy_fine,
                    rec_key * key_map[key]["factor"],
                    color=POWER_LAW_FIT_COLOR,
                    linestyle="-",
                )

            if PLOT_TITLE_INFO:
                ax.set_title(info_str, alpha=0.5)
            ax.semilogx()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel("energy$\,/\,$GeV")
            ax.set_xlim([0.4, 110])

            y_fit_lower = key_map[key]["factor"] * np.min(unc80_lower)
            y_fit_upper = key_map[key]["factor"] * np.max(unc80_upper)
            y_fit_range = y_fit_upper - y_fit_lower
            assert y_fit_range >= 0
            y_fit_range = np.max([y_fit_range, 1.0])
            _ll = y_fit_lower - 0.2 * y_fit_range
            _uu = y_fit_upper + 0.2 * y_fit_range
            ax.set_ylim([_ll, _uu])
            ax.set_ylabel(
                "{key:s}$\,/\,${unit:s}".format(
                    key=key_map[key]["name"], unit=key_map[key]["unit"]
                )
            )
            ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, key)
            filepath = os.path.join(deflection_dir, filename)
            fig.savefig(filepath + ".jpg")
            plt.close(fig)

            _fit = {
                "name": key,
                "power_law": {
                    "formula": "f(Energy) = A*Energy**B + C",
                    "A": float(expy[0]),
                    "B": float(expy[1]),
                    "C": float(key_start),
                },
                "particle_energy_GeV": sres["energy_supports"].tolist(),
                "mean": sres["key_mean80"].tolist(),
                "std": sres["key_std80"].tolist(),
            }

            with open(filepath + ".json", "wt") as fout:
                fout.write(json.dumps(_fit, indent=4))

            power_law_fit_table[skey][pkey][key] = _fit

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
        mdfl_plot.add_grid_in_half_dome(
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
        mdfl_plot.add_points_in_half_dome(
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
        mdfl_plot.add_ticklabels_in_half_dome(
            ax=ax, azimuths_deg=azimuths_deg_steps, rfov=rfov
        )
        ax.set_xlim([-1.01 * rfov, 1.01 * rfov])
        ax.set_ylim([-1.01 * rfov, 1.01 * rfov])
        fig.savefig(
            os.path.join(
                deflection_dir,
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
            os.path.join(deflection_dir, "{:s}_{:s}.jpg".format(skey, den_key))
        )
        plt.close(fig)


_table = make_latex_table_with_power_law_fit(power_law_fit_table)
with open(os.path.join(deflection_dir, "power_law_table.tex"), "wt") as fout:
    fout.write(_table)


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
fig.savefig(os.path.join(deflection_dir, "{:s}.jpg".format(den_key)))
plt.close(fig)
