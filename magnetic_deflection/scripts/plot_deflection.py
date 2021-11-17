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

matplotlib.rcParams["mathtext.fontset"] = PLT["rcParams"]["mathtext.fontset"]
matplotlib.rcParams["font.family"] = PLT["rcParams"]["font.family"]

FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}
AXSPAN = [0.15, 0.2, 0.8, 0.75]

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
    "position_med_x_m": {
        "unit": "km",
        "name": "Cherenkov-pool-x",
        "factor": 1e-3,
    },
    "position_med_y_m": {
        "unit": "km",
        "name": "Cherenkov-pool-y",
        "factor": 1e-3,
    },
}

defl = {}
for skey in CFG["sites"]:
    defl[skey] = {}
    for pkey in CFG["particles"]:
        d = {}
        ddir = os.path.join(work_dir, "reduce", skey, pkey, ".deflection")
        d["raw"] = mdfl.recarray_io.read_from_csv(os.path.join(ddir, "raw.csv"))
        d["raw_valid"] = mdfl.recarray_io.read_from_csv(os.path.join(ddir, "raw_valid.csv"))
        d["raw_valid_add"] = mdfl.recarray_io.read_from_csv(os.path.join(ddir, "raw_valid_add.csv"))
        d["raw_valid_add_clean"] = mdfl.recarray_io.read_from_csv(os.path.join(ddir, "raw_valid_add_clean.csv"))
        d["raw_valid_add_clean_high"] = mdfl.recarray_io.read_from_csv(os.path.join(ddir, "raw_valid_add_clean_high.csv"))
        d["raw_valid_add_clean_high_power"] = mdfl.tools.read_json(os.path.join(ddir, "raw_valid_add_clean_high_power.json"))
        d["result"] = mdfl.recarray_io.read_from_csv(
            os.path.join(
                work_dir, "reduce", skey, pkey, "deflection.csv"
            )
        )
        defl[skey][pkey] = d

for skey in CFG["sites"]:
    for pkey in CFG["particles"]:
        for key in mdfl.analysis.FIT_KEYS:

            fig = sebplt.figure(FIGSIZE)
            ax = sebplt.add_axes(fig, AXSPAN)

            ax.plot(
                defl[skey][pkey]["raw"]["particle_energy_GeV"],
                defl[skey][pkey]["raw"][key] * key_map[key]["factor"],
                "ko",
                alpha=0.05,
            )

            ax.plot(
                defl[skey][pkey]["raw_valid_add_clean"]["particle_energy_GeV"],
                defl[skey][pkey]["raw_valid_add_clean"][key] * key_map[key]["factor"],
                "kx",
            )
            num_e = len(defl[skey][pkey]["raw_valid_add_clean"]["particle_energy_GeV"])
            for ibin in range(num_e):
                _x = defl[skey][pkey]["raw_valid_add_clean"]["particle_energy_GeV"][
                    ibin
                ]
                _y_std = defl[skey][pkey]["raw_valid_add_clean"][key + "_std"][ibin]
                _y = defl[skey][pkey]["raw_valid_add_clean"][key][ibin]
                _y_low = _y - _y_std
                _y_high = _y + _y_std
                ax.plot(
                    [_x, _x],
                    np.array([_y_low, _y_high]) * key_map[key]["factor"],
                    "k-",
                )

            ax.plot(
                defl[skey][pkey]["raw_valid_add_clean_high"]["particle_energy_GeV"],
                defl[skey][pkey]["raw_valid_add_clean_high"][key]
                * key_map[key]["factor"],
                "bo",
                alpha=0.3,
            )

            ax.plot(
                defl[skey][pkey]["result"]["particle_energy_GeV"],
                defl[skey][pkey]["result"][key] * key_map[key]["factor"],
                color="k",
                linestyle="-",
            )

            ax.semilogx()
            ax.set_xlabel("energy$\,/\,$GeV")
            ax.set_xlim([0.1, 100])

            ax.set_ylabel(
                "{key:s}$\,/\,${unit:s}".format(
                    key=key_map[key]["name"], unit=key_map[key]["unit"]
                )
            )
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, key)
            filepath = os.path.join(out_dir, filename)
            fig.savefig(filepath + ".jpg")
            sebplt.close_figure(fig)
