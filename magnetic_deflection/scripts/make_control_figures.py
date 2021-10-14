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


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]

figsize = (16 / 2, 9 / 2)
dpi = 240
ax_size = (0.15, 0.12, 0.8, 0.8)

key_map = {
    "particle_azimuth_deg": {
        "unit": "deg",
        "name": "particle-azimuth",
        "factor": 1,
    },
    "particle_zenith_deg": {
        "unit": "deg",
        "name": "particle-zenith",
        "factor": 1,
    },
    "cherenkov_pool_x_m": {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
    },
    "cherenkov_pool_y_m": {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
    },
}

with open(os.path.join(work_dir, "sites.json"), "rt") as f:
    sites = json.loads(f.read())
with open(os.path.join(work_dir, "particles.json"), "rt") as f:
    particles = json.loads(f.read())
with open(os.path.join(work_dir, "pointing.json"), "rt") as f:
    pointing = json.loads(f.read())

raw = mdfl.tools.read_deflection_table(path=os.path.join(work_dir, "raw"))

raw_valid_add_clean = mdfl.tools.read_deflection_table(
    path=os.path.join(work_dir, "raw_valid_add_clean")
)

raw_valid_add_clean_high = mdfl.tools.read_deflection_table(
    path=os.path.join(work_dir, "raw_valid_add_clean_high")
)

result = mdfl.tools.read_deflection_table(
    path=os.path.join(work_dir, "result")
)

out_dir = os.path.join(work_dir, "control_figures")
os.makedirs(out_dir, exist_ok=True)

for skey in sites:
    if "Off" in skey:
        continue
    for pkey in particles:
        for key in mdfl.FIT_KEYS:

            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes(ax_size)

            ax.plot(
                raw[skey][pkey]["particle_energy_GeV"],
                raw[skey][pkey][key] * key_map[key]["factor"],
                "ko",
                alpha=0.05,
            )

            ax.plot(
                raw_valid_add_clean[skey][pkey]["particle_energy_GeV"],
                raw_valid_add_clean[skey][pkey][key] * key_map[key]["factor"],
                "kx",
            )
            num_e = len(raw_valid_add_clean[skey][pkey]["particle_energy_GeV"])
            for ibin in range(num_e):
                _x = raw_valid_add_clean[skey][pkey]["particle_energy_GeV"][
                    ibin
                ]
                _y_std = raw_valid_add_clean[skey][pkey][key + "_std"][ibin]
                _y = raw_valid_add_clean[skey][pkey][key][ibin]
                _y_low = _y - _y_std
                _y_high = _y + _y_std
                ax.plot(
                    [_x, _x],
                    np.array([_y_low, _y_high]) * key_map[key]["factor"],
                    "k-",
                )

            ax.plot(
                raw_valid_add_clean_high[skey][pkey]["particle_energy_GeV"],
                raw_valid_add_clean_high[skey][pkey][key]
                * key_map[key]["factor"],
                "bo",
                alpha=0.3,
            )

            ax.plot(
                result[skey][pkey]["particle_energy_GeV"],
                result[skey][pkey][key] * key_map[key]["factor"],
                color="k",
                linestyle="-",
            )

            ax.semilogx()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel("energy$\,/\,$GeV")
            ax.set_xlim([0.4, 110])

            ax.set_ylabel(
                "{key:s}$\,/\,${unit:s}".format(
                    key=key_map[key]["name"], unit=key_map[key]["unit"]
                )
            )
            ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, key)
            filepath = os.path.join(out_dir, filename)
            fig.savefig(filepath + ".jpg")
            plt.close(fig)
