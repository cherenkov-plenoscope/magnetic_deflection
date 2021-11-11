#!/usr/bin/python
import sys
import os
import numpy as np
import json
import scipy
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot_deflection_table")
os.makedirs(out_dir, exist_ok=True)


CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]

matplotlib.rcParams["mathtext.fontset"] = PLT["rcParams"]["mathtext.fontset"]
matplotlib.rcParams["font.family"] = PLT["rcParams"]["font.family"]

FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}
FIGSIZE_FIT = {"rows": 720, "cols": 1280, "fontsize": 1.25}
AXSPAN = [0.15, 0.2, 0.8, 0.75]

PLOT_TITLE_INFO = False

PLOT_POWER_LAW_FIT = True
POWER_LAW_FIT_COLOR = "k"
PLOT_ENERGY_SUPPORTS = False
PLOT_RAW_ESTIMATES = True

key_map = {
    "particle_azimuth_deg": {
        "unit": "(1$^\\circ$)",
        "name": "particle-azimuth",
        "start": 90.0,
        "extend_high_energies": True,
        "latex_key": "\\PrmAz{}",
    },
    "particle_zenith_deg": {
        "unit": "(1$^\\circ$)",
        "name": "particle-zenith",
        "start": 0.0,
        "extend_high_energies": True,
        "latex_key": "\\PrmZd{}",
    },
    "position_med_x_m": {
        "unit": "m",
        "name": "Cherenkov-pool-x",
        "start": 0.0,
        "extend_high_energies": True,
        "latex_key": "\\CerX{}",
    },
    "position_med_y_m": {
        "unit": "m",
        "name": "Cherenkov-pool-y",
        "start": 0.0,
        "extend_high_energies": True,
        "latex_key": "\\CerY{}",
    },
}

fit_map = {
    "A": {"latex_key": "$\\PowerLawA{}$"},
    "B": {"latex_key": "$\\PowerLawB{}$"},
    "C": {"latex_key": "$\\PowerLawC{}$"},
}


charge_signs = {}
for pkey in CFG["particles"]:
    charge_signs[pkey] = np.sign(
        CFG["particles"][pkey]["electric_charge_qe"]
    )


def make_site_str(skey, site):
    return "".join(
        [
            "{:s}, {:.1f}$\\,$km$\\,$a.s.l., ",
            "Atm.-id {:d}, ",
            "Bx {:.1f}$\\,$uT, ",
            "Bz {:.1f}$\\,$uT",
        ]
    ).format(
        skey,
        site["observation_level_asl_m"] * 1e-3,
        site["atmosphere_id"],
        site["earth_magnetic_field_x_muT"],
        site["earth_magnetic_field_z_muT"],
    )


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


def make_latex_table_with_power_law_fit(power_law_fit_table):
    matrix = []
    partile_line = ["", "", ""]
    for pkey in PLT["particles"]:
        partile_line.append(PLT["particles"][pkey]["label"])
    matrix.append(partile_line)
    for skey in power_law_fit_table:
        if "Off" in skey:
            continue
        site_line = [
            PLT["sites"][skey]["label"],
            "",
            "",
            "",
            "",
            "",
            "",
        ]
        matrix.append(site_line)
        for variable_key in key_map:
            variable_line = [
                "",
                key_map[variable_key]["latex_key"]
                + "\\,/\\,"
                + key_map[variable_key]["unit"],
                "",
                "",
                "",
                "",
                "",
            ]
            matrix.append(variable_line)

            for param_key in fit_map:
                value_line = ["", "", fit_map[param_key]["latex_key"]]
                for pkey in power_law_fit_table[skey]:

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


deflection_table = mdfl.tools.read_deflection_table(
    os.path.join(work_dir, "raw")
)

deflection_table = mdfl.analysis.cut_invalid_from_deflection_table(
    deflection_table=deflection_table, but_keep_site="Off"
)
deflection_table = mdfl.analysis.add_density_fields_to_deflection_table(
    deflection_table=deflection_table
)

power_law_fit_table = {}
for skey in deflection_table:
    power_law_fit_table[skey] = {}
    for pkey in deflection_table[skey]:
        print(skey, pkey)
        site_str = make_site_str(skey, CFG["sites"][skey])

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

            if key_map[key]["extend_high_energies"]:
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

            fig = sebplt.figure(FIGSIZE_FIT)
            ax = sebplt.add_axes(fig, AXSPAN)
            if PLOT_RAW_ESTIMATES:
                ax.plot(
                    t["particle_energy_GeV"],
                    np.array(t[key]),
                    "ko",
                    alpha=0.05,
                )

            if PLOT_ENERGY_SUPPORTS:
                ax.plot(energy_supports, key_mean80, "kx")
                for ibin in range(len(energy_supports)):
                    _x = energy_supports[ibin]
                    _y_low = unc80_lower[ibin]
                    _y_high = unc80_upper[ibin]
                    ax.plot(
                        [_x, _x], np.array([_y_low, _y_high]), "k-",
                    )
                ax.plot(
                    energy_bins_ext, key_mean80_ext, "bo", alpha=0.3,
                )

            if PLOT_POWER_LAW_FIT:
                ax.plot(
                    energy_fine,
                    rec_key,
                    color=POWER_LAW_FIT_COLOR,
                    linestyle="-",
                )

            if PLOT_TITLE_INFO:
                ax.set_title(info_str, alpha=0.5)
            ax.semilogx()
            ax.set_xlabel("energy"+PLT["label_unit_seperator"]+"GeV")
            ax.set_xlim([0.4, 110])

            y_fit_lower = np.min(unc80_lower)
            y_fit_upper = np.max(unc80_upper)
            y_fit_range = y_fit_upper - y_fit_lower
            assert y_fit_range >= 0
            y_fit_range = np.max([y_fit_range, 1.0])
            _ll = y_fit_lower - 0.2 * y_fit_range
            _uu = y_fit_upper + 0.2 * y_fit_range
            ax.set_ylim([_ll, _uu])
            ax.set_ylabel(
                key_map[key]["name"]
                + PLT["label_unit_seperator"]
                + key_map[key]["unit"]
            )
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, key)
            filepath = os.path.join(out_dir, filename)
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

    for dkey in PLT["light_field"]:
        ts = deflection_table[skey]
        alpha = 0.2
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig, AXSPAN)
        for pkey in ts:
            ax.plot(
                ts[pkey]["particle_energy_GeV"],
                ts[pkey][dkey],
                "o",
                color=PLT["particles"][pkey]["color"],
                alpha=alpha,
                label=PLT["particles"][pkey]["label"],
            )
        leg = ax.legend()
        if PLOT_TITLE_INFO:
            ax.set_title(site_str, alpha=0.5)
        ax.loglog()
        ax.set_xlabel("energy"+PLT["label_unit_seperator"]+"GeV")
        ax.set_xlim([1e-1, 1e2])
        ax.set_ylim(PLT["light_field"][dkey]["limits"])
        ax.set_ylabel(
            PLT["light_field"][dkey]["label"]
            + PLT["label_unit_seperator"]
            + PLT["light_field"][dkey]["unit"]
        )
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(skey, dkey)))
        plt.close(fig)

_table = make_latex_table_with_power_law_fit(power_law_fit_table)
with open(os.path.join(out_dir, "power_law_table.tex"), "wt") as fout:
    fout.write(_table)

# density side by side
# --------------------

dkey = "light_field_outer_density"


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


alpha = 0.2
fig = sebplt.figure(FIGSIZE)
ax = sebplt.add_axes(fig, AXSPAN)
for pkey in ["electron", "gamma"]:
    particle_color = PLT["particles"][pkey]["color"]
    for skey in deflection_table:
        E = deflection_table[skey][pkey]["particle_energy_GeV"]
        V = deflection_table[skey][pkey][dkey]
        mask = np.arange(20, len(E), len(E) // 10)
        ax.plot(
            E,
            V,
            PLT["sites"][skey]["marker"],
            color=particle_color,
            alpha=0.1 * alpha,
        )
        if particle_color == "black":
            label = PLT["sites"][skey]["label"]
        else:
            label = None
        ax.plot(
            E[mask],
            smooth(V, 9)[mask],
            PLT["sites"][skey]["marker"],
            color=particle_color,
            label=label,
        )

ax.text(
    1.1,
    50,
    PLT["particles"]["gamma"]["label"],
    color=PLT["particles"]["gamma"]["color"],
)
ax.text(
    1.1,
    25,
    PLT["particles"]["electron"]["label"],
    color=PLT["particles"]["electron"]["color"],
)
leg = ax.legend()
ax.loglog()
ax.set_xlabel("energy"+PLT["label_unit_seperator"]+"GeV")
ax.set_xlim([1e-1, 1e2])
ax.set_ylim([1e-3, 1e2])
ax.set_ylabel(
    PLT["light_field"][dkey]["label"]
    + PLT["label_unit_seperator"]
    + PLT["light_field"][dkey]["unit"]
)
fig.savefig(os.path.join(out_dir, "{:s}.jpg".format(dkey)))
plt.close(fig)
