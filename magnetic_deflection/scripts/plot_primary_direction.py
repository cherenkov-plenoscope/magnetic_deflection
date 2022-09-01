#!/usr/bin/python
import sys
import os
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt

print("start")
argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "primary_direction")
os.makedirs(out_dir, exist_ok=True)

CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 2}
CMAP_FIGSIZE = {"rows": 300, "cols": 1280, "fontsize": 1.75}

ON_AXIS_SCALE = 1.0

HEMISPHERE_AXSTYLE = {"spines": [], "axes": [], "grid": False}

ENERGY_START_GEV = 0.1
ENERGY_STOP_GEV = 100

FRACTION = 1.0

# energy colorbar
# ---------------

cmap_fig = sebplt.figure(CMAP_FIGSIZE)
cmap_ax = sebplt.add_axes(
    fig=cmap_fig, span=(0.05, 0.75, 0.9, 0.2), style=sebplt.AXES_MATPLOTLIB
)
cmap_name = "nipy_spectral"
cmap_norm = plt_colors.LogNorm(vmin=ENERGY_START_GEV, vmax=ENERGY_STOP_GEV,)
cmap_mappable = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap_name)
plt.colorbar(cmap_mappable, cax=cmap_ax, orientation="horizontal")
cmap_ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
cmap_fig.savefig(os.path.join(out_dir, "energy_colorbar.jpg"))
sebplt.close(cmap_fig)

shower_statistics = mdfl.read_statistics(work_dir=work_dir)

for skey in shower_statistics:
    for pkey in shower_statistics[skey]:
        sort_args = np.argsort(
            shower_statistics[skey][pkey]["particle_energy_GeV"]
        )
        shower_statistics[skey][pkey] = shower_statistics[skey][pkey][
            sort_args
        ]

prng = np.random.generator.Generator(np.random.generator.PCG64(1337))

# hemisphere showing deflections
# ------------------------------
FIELD_OF_VIEW = {
    "wide": {
        "angle_deg": 60,
        "particles": list(CFG["particles"].keys()),
        "zenith_mayor_deg": [0, 20, 40, 60],
        "zenith_minor_deg": [0, 10, 20, 30, 40, 40, 50, 60],
    },
    "narrow": {
        "angle_deg": 10,
        "particles": ["gamma"],
        "zenith_mayor_deg": [0, 2, 4, 6, 8, 10],
        "zenith_minor_deg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
}

ALPHA = 0.5

for skey in shower_statistics:
    for pkey in shower_statistics[skey]:

        showers = shower_statistics[skey][pkey]

        for fkey in FIELD_OF_VIEW:
            if pkey not in FIELD_OF_VIEW[fkey]["particles"]:
                continue

            fov_deg = FIELD_OF_VIEW[fkey]["angle_deg"]
            azimuth_mayor_deg = np.linspace(0, 360, 12, endpoint=False)
            zenith_mayor_deg = FIELD_OF_VIEW[fkey]["zenith_mayor_deg"]
            azimuth_minor_deg = np.linspace(0, 360, 24, endpoint=False)
            zenith_minor_deg = FIELD_OF_VIEW[fkey]["zenith_minor_deg"]
            rfov = np.sin(np.deg2rad(fov_deg))

            print(skey, pkey, fkey)

            fig = sebplt.figure(FIGSIZE)
            ax = sebplt.add_axes(
                fig=fig,
                span=(0.02, 0.02, 0.96, 0.96),
                style=HEMISPHERE_AXSTYLE,
            )
            # mayor
            sebplt.hemisphere.ax_add_grid(
                ax=ax,
                azimuths_deg=azimuth_mayor_deg,
                zeniths_deg=zenith_mayor_deg,
                linewidth=1.4,
                color="k",
                alpha=0.1,
                draw_lower_horizontal_edge_deg=fov_deg,
            )
            # minor
            sebplt.hemisphere.ax_add_grid(
                ax=ax,
                azimuths_deg=azimuth_minor_deg,
                zeniths_deg=zenith_minor_deg,
                linewidth=1.4 * 0.5,
                color="k",
                alpha=0.1,
                draw_lower_horizontal_edge_deg=None,
            )

            mask_on_axis = (
                showers["off_axis_deg"]
                <= ON_AXIS_SCALE
                * CFG["particles"][pkey][
                    "magnetic_deflection_max_off_axis_deg"
                ]
            )

            rgbas = cmap_mappable.to_rgba(
                showers[mask_on_axis]["particle_energy_GeV"]
            )
            rgbas[:, 3] = ALPHA

            _fm = prng.uniform(size=rgbas.shape[0]) <= FRACTION

            sebplt.hemisphere.ax_add_points(
                ax=ax,
                azimuths_deg=showers[mask_on_axis]["particle_azimuth_deg"][
                    _fm
                ],
                zeniths_deg=showers[mask_on_axis]["particle_zenith_deg"][_fm],
                point_diameter_deg=CFG["particles"][pkey][
                    "magnetic_deflection_max_off_axis_deg"
                ],
                rgbas=rgbas[_fm],
            )

            ax.text(
                -1.0 * rfov, -1.0 * rfov, "{:1.1f}$^\\circ$".format(fov_deg)
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            sebplt.hemisphere.ax_add_ticklabels(
                ax=ax, azimuths_deg=[0, 90, 180, 270], rfov=0.93 * rfov
            )
            ax.set_xlim([-1.01 * rfov, 1.01 * rfov])
            ax.set_ylim([-1.01 * rfov, 1.01 * rfov])
            fig.savefig(
                os.path.join(
                    out_dir, "{:s}_{:s}_{:s}.jpg".format(skey, pkey, fkey),
                )
            )
            sebplt.close(fig)
