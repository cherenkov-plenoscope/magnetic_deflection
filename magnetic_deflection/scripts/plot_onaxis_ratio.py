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


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "onaxis_ratio")
os.makedirs(out_dir, exist_ok=True)

CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]

matplotlib.rcParams["mathtext.fontset"] = PLT["rcParams"]["mathtext.fontset"]
matplotlib.rcParams["font.family"] = PLT["rcParams"]["font.family"]
FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}

MIN_NUM_SHOWER = 11
ON_AXIS_SCALE = 1.0

ENERGY = {}
ENERGY["num_bins"] = 15
ENERGY["bin_edges"] = np.geomspace(1e-1, 1e2, ENERGY["num_bins"] + 1)

shower_statistics = mdfl.read_statistics(work_dir=work_dir)

instrument_azimuth_deg = CFG["pointing"]["azimuth_deg"]
instrument_zenith_deg = CFG["pointing"]["zenith_deg"]

# cut on-axis
# -----------
counts = {}
for skey in shower_statistics:
    counts[skey] = {}
    for pkey in shower_statistics[skey]:
        counts[skey][pkey] = {}
        sst = shower_statistics[skey][pkey]
        counts[skey][pkey] = {
            "num_particles_onaxis": np.zeros(ENERGY["num_bins"]),
            "num_cherenkov_onaxis": np.zeros(ENERGY["num_bins"]),
            "num_particles_and_cherenkov_onaxis": np.zeros(ENERGY["num_bins"]),
        }

        particle_off_axis_deg = mdfl.spherical_coordinates._angle_between_az_zd_deg(
            az1_deg=sst["particle_azimuth_deg"],
            zd1_deg=sst["particle_zenith_deg"],
            az2_deg=instrument_azimuth_deg,
            zd2_deg=instrument_zenith_deg,
        )
        mask_particle_onaxis = (
            particle_off_axis_deg
            <= ON_AXIS_SCALE
            * CFG["particles"][pkey]["magnetic_deflection_max_off_axis_deg"]
        )

        (
            cer_azimuth_deg,
            cer_zenith_deg,
        ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
            cx=sst["cherenkov_cx_rad"], cy=sst["cherenkov_cy_rad"],
        )

        cherenkov_off_axis_deg = mdfl.spherical_coordinates._angle_between_az_zd_deg(
            az1_deg=cer_azimuth_deg,
            zd1_deg=cer_zenith_deg,
            az2_deg=instrument_azimuth_deg,
            zd2_deg=instrument_zenith_deg,
        )
        mask_cherenkov_onaxis = (
            cherenkov_off_axis_deg
            <= ON_AXIS_SCALE
            * CFG["particles"][pkey]["magnetic_deflection_max_off_axis_deg"]
        )

        for ebin in range(ENERGY["num_bins"]):
            print(skey, pkey, ebin)
            energy_start = ENERGY["bin_edges"][ebin]
            energy_stop = ENERGY["bin_edges"][ebin + 1]

            mask_energy = np.logical_and(
                sst["particle_energy_GeV"] >= energy_start,
                sst["particle_energy_GeV"] < energy_stop,
            )

            mask_ene_par = np.logical_and(mask_energy, mask_particle_onaxis)
            mask_ene_cer = np.logical_and(mask_energy, mask_cherenkov_onaxis)
            mask_ene_par_cer = np.logical_and(mask_ene_par, mask_ene_cer)

            counts[skey][pkey]["num_particles_onaxis"][ebin] = np.sum(
                mask_ene_par
            )
            counts[skey][pkey]["num_cherenkov_onaxis"][ebin] = np.sum(
                mask_ene_cer
            )
            counts[skey][pkey]["num_particles_and_cherenkov_onaxis"][
                ebin
            ] = np.sum(mask_ene_par_cer)

alpha = 0.2
for skey in counts:
    fig = sebplt.figure(FIGSIZE)
    ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

    for pkey in counts[skey]:
        for ebin in range(ENERGY["num_bins"]):
            energy_start = ENERGY["bin_edges"][ebin]
            energy_stop = ENERGY["bin_edges"][ebin + 1]

            if (
                counts[skey][pkey]["num_particles_onaxis"][ebin]
                >= MIN_NUM_SHOWER
            ):
                ratio = (
                    counts[skey][pkey]["num_particles_and_cherenkov_onaxis"][
                        ebin
                    ]
                    / counts[skey][pkey]["num_particles_onaxis"][ebin]
                )
                n = np.sqrt(
                    counts[skey][pkey]["num_particles_and_cherenkov_onaxis"][
                        ebin
                    ]
                )
                rel_unc_ratio = np.sqrt(n) / n
            else:
                ratio = np.nan
                rel_unc_ratio = np.nan

            ax.plot(
                [energy_start, energy_stop],
                [ratio, ratio],
                color=PLT["particles"][pkey]["color"],
                alpha=1.0,
            )
            ratio_upper = ratio * (1 + rel_unc_ratio)
            ratio_lower = ratio * (1 - rel_unc_ratio)
            ax.fill(
                [energy_start, energy_start, energy_stop, energy_stop],
                [ratio_lower, ratio_upper, ratio_upper, ratio_lower],
                color=PLT["particles"][pkey]["color"],
                alpha=alpha,
                linewidth=0.0,
            )

    ax.semilogx()
    ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
    ax.set_ylabel(
        "#(par. and Cher.) / #(par.)" + PLT["label_unit_seperator"] + "1"
    )
    ax.set_xlim([min(ENERGY["bin_edges"]), max(ENERGY["bin_edges"])])
    ax.set_ylim([0, 1.05])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(os.path.join(out_dir, "{:s}_ratio.jpg".format(skey)))
    sebplt.close(fig)
