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


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot_primary_direction")
os.makedirs(out_dir, exist_ok=True)

shower_statistics = mdfl.read_shower_statistics(work_dir=work_dir)

c = mdfl.read_config(work_dir=work_dir)

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"

figsize = {"rows": 1280, "cols": 1280, "fontsize": 1.5}
cmap_figsize = {"rows": 240, "cols": 1280, "fontsize": 1.5}

ON_AXIS_SCALE = 1.0

hemisphere_axstyle = {"spines": [], "axes": [], "grid": False}

energy_start_GeV = 0.1
energy_stop_GeV = 100

# energy colorbar
# ---------------

cmap_fig = sebplt.figure(cmap_figsize)
cmap_ax = sebplt.add_axes(
    fig=cmap_fig, span=(0.05, 0.75, 0.9, 0.2), style=sebplt.AXES_MATPLOTLIB
)
cmap_name = "nipy_spectral"
cmap_norm = plt_colors.LogNorm(vmin=energy_start_GeV, vmax=energy_stop_GeV,)
cmap_mappable = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap_name)
plt.colorbar(cmap_mappable, cax=cmap_ax, orientation="horizontal")
cmap_ax.set_xlabel("energy$\,/\,$GeV")
cmap_fig.savefig(os.path.join(out_dir, "energy_colorbar.jpg"))
plt.close(cmap_fig)


# hemisphere showing deflections
# ------------------------------
field_of_view = {
    "wide": {
        "angle_deg": 45,
        "particles": list(c["particles"].keys()),
        "zenith_mayor_deg": [0, 10, 20, 30, 40, 45,],
        "zenith_minor_deg": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,],
    },
    "narrow": {
        "angle_deg": 5,
        "particles": ["gamma"],
        "zenith_mayor_deg": [0, 1, 2, 3, 4, 5,],
        "zenith_minor_deg": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,]
    },
}

for skey in shower_statistics:
    for pkey in shower_statistics[skey]:

        showers = shower_statistics[skey][pkey]

        for fkey in field_of_view:
            if pkey not in field_of_view[fkey]["particles"]:
                continue

            fov_deg = field_of_view[fkey]["angle_deg"]
            azimuth_mayor_deg = np.linspace(0, 360, 12, endpoint=False)
            zenith_mayor_deg = field_of_view[fkey]["zenith_mayor_deg"]
            azimuth_minor_deg = np.linspace(0, 360, 24, endpoint=False)
            zenith_minor_deg = field_of_view[fkey]["zenith_minor_deg"]
            fov = np.deg2rad(fov_deg)
            rfov = np.sin(fov)

            print(skey, pkey, fkey)

            fig = sebplt.figure(figsize)
            ax = sebplt.add_axes(
                fig=fig,
                span=(0.02, 0.02, 0.96, 0.96),
                style=hemisphere_axstyle,
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
                * c["particles"][pkey]["magnetic_deflection_max_off_axis_deg"]
            )

            rgbas = cmap_mappable.to_rgba(
                showers[mask_on_axis]["particle_energy_GeV"]
            )
            rgbas[:, 3] = 0.5

            sebplt.hemisphere.ax_add_points(
                ax=ax,
                azimuths_deg=showers[mask_on_axis]["particle_azimuth_deg"],
                zeniths_deg=showers[mask_on_axis]["particle_zenith_deg"],
                point_diameter_deg=c["particles"][pkey][
                    "magnetic_deflection_max_off_axis_deg"
                ],
                rgbas=rgbas,
            )

            ax.text(
                -1.0 * rfov, -1.0 * rfov, "{:1.0f}$^\circ$".format(fov_deg)
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
            plt.close(fig)
