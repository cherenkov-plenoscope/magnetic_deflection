#!/usr/bin/python
import sys
import os
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "deflection")
os.makedirs(out_dir, exist_ok=True)

CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 2}
AXSPAN = [0.25, 0.17, 0.7, 0.75]

PLOT_RAW = False
PLOT_RAW_VALID_ADD_CLEAN = True
PLOT_RAW_VALID_ADD_CLEAN_HIGH = True
PLOT_FIT_RESULT = True
DEG_UNIT_LATEX_STR = "$1^{\\circ}$"

key_map = {
    "particle_azimuth_deg": {
        "unit": DEG_UNIT_LATEX_STR,
        "name": "particle-azimuth",
        "factor": 1,
        "ylim": [-180, 180],
    },
    "particle_zenith_deg": {
        "unit": DEG_UNIT_LATEX_STR,
        "name": "particle-zenith",
        "factor": 1,
        "ylim": [0, 70],
    },
    "cherenkov_x_m": {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
        "ylim": [-200, 200],
    },
    "cherenkov_y_m": {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
        "ylim": [-200, 200],
    },
}

defl = {}
for skey in CFG["sites"]:
    defl[skey] = {}
    for pkey in CFG["particles"]:
        d = {}
        ddir = os.path.join(work_dir, "reduce", skey, pkey, ".deflection")
        d["raw"] = mdfl.recarray_io.read_from_csv(
            os.path.join(ddir, "raw.csv")
        )
        d["raw_valid"] = mdfl.recarray_io.read_from_csv(
            os.path.join(ddir, "raw_valid.csv")
        )
        d["raw_valid_add"] = mdfl.recarray_io.read_from_csv(
            os.path.join(ddir, "raw_valid_add.csv")
        )
        d["raw_valid_add_clean"] = mdfl.recarray_io.read_from_csv(
            os.path.join(ddir, "raw_valid_add_clean.csv")
        )
        d["raw_valid_add_clean_high"] = mdfl.recarray_io.read_from_csv(
            os.path.join(ddir, "raw_valid_add_clean_high.csv")
        )
        d["raw_valid_add_clean_high_power"] = mdfl.tools.read_json(
            os.path.join(ddir, "raw_valid_add_clean_high_power.json")
        )
        d["result"] = mdfl.recarray_io.read_from_csv(
            os.path.join(work_dir, "reduce", skey, pkey, "deflection.csv")
        )
        defl[skey][pkey] = d

for skey in CFG["sites"]:
    for pkey in CFG["particles"]:
        for key in mdfl.analysis.FIT_KEYS:

            fig = sebplt.figure(FIGSIZE)
            ax = sebplt.add_axes(fig, AXSPAN)

            if PLOT_RAW:
                ax.plot(
                    defl[skey][pkey]["raw"]["particle_energy_GeV"],
                    defl[skey][pkey]["raw"][key] * key_map[key]["factor"],
                    "ko",
                    alpha=0.05,
                )

            if PLOT_RAW_VALID_ADD_CLEAN:
                if False:
                    ax.plot(
                        defl[skey][pkey]["raw_valid_add_clean"][
                            "particle_energy_GeV"
                        ],
                        defl[skey][pkey]["raw_valid_add_clean"][key]
                        * key_map[key]["factor"],
                        "kx",
                        alpha=0.3,
                    )
                num_e = len(
                    defl[skey][pkey]["raw_valid_add_clean"][
                        "particle_energy_GeV"
                    ]
                )
                for ibin in range(num_e):
                    _x = defl[skey][pkey]["raw_valid_add_clean"][
                        "particle_energy_GeV"
                    ][ibin]
                    _y_std = defl[skey][pkey]["raw_valid_add_clean"][
                        key + "_std"
                    ][ibin]
                    _y = defl[skey][pkey]["raw_valid_add_clean"][key][ibin]
                    _y_low = _y - _y_std
                    _y_high = _y + _y_std
                    ax.plot(
                        [_x, _x],
                        np.array([_y_low, _y_high]) * key_map[key]["factor"],
                        "k-",
                        alpha=0.3,
                    )
            if PLOT_RAW_VALID_ADD_CLEAN_HIGH:
                ax.plot(
                    defl[skey][pkey]["raw_valid_add_clean_high"][
                        "particle_energy_GeV"
                    ],
                    defl[skey][pkey]["raw_valid_add_clean_high"][key]
                    * key_map[key]["factor"],
                    "ko",
                    alpha=0.3,
                )

            if PLOT_FIT_RESULT:
                ax.plot(
                    defl[skey][pkey]["result"]["particle_energy_GeV"],
                    defl[skey][pkey]["result"][key] * key_map[key]["factor"],
                    color="k",
                    linestyle="-",
                )

            ax.semilogx()
            ax.set_ylim(key_map[key]["ylim"])
            ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
            ax.set_xlim([0.1, 100])

            ax.set_ylabel(
                "{key:s}{sep:s}{unit:s}".format(
                    key=key_map[key]["name"],
                    sep=PLT["label_unit_seperator"],
                    unit=key_map[key]["unit"],
                )
            )
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, key)
            filepath = os.path.join(out_dir, filename)
            fig.savefig(filepath + ".jpg")
            sebplt.close(fig)
