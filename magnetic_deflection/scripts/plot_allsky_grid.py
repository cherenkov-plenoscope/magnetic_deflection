import argparse
import os
import numpy as np
import magnetic_deflection as mdfl
import binning_utils
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import importlib
from importlib import resources
import triangle_mesh_io


parser = argparse.ArgumentParser(
    prog="plot_allsky_cherenkov_density.py",
    description=("Make plots of Cherenkov statistics"),
)
parser.add_argument(
    "--work_dir",
    metavar="STRING",
    type=str,
    help=(
        "Work_dir with sites. "
        "The site directories contain one AllSky for each particle."
    ),
)
parser.add_argument(
    "--out_dir",
    metavar="STRING",
    type=str,
    help="Directory to write figures to.",
)
parser.add_argument(
    "--half_angle_deg",
    metavar="FLOAT",
    type=float,
    default=10.0,
    help="Half angle of instrument's field-of-view.",
)

args = parser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
half_angle_deg = args.half_angle_deg

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 2}
CMAP_FIGSIZE = {"rows": 300, "cols": 1280, "fontsize": 1.75}

ON_AXIS_SCALE = 1.0

HEMISPHERE_AXSTYLE = {"spines": [], "axes": [], "grid": False}

ENERGY_START_GEV = 0.5
ENERGY_STOP_GEV = 100
GRID_COLOR = (0.5, 0.5, 0.5)

ALPHA = 0.75

# energy colorbar
# ---------------

cmap_fig = sebplt.figure(CMAP_FIGSIZE)
cmap_ax = sebplt.add_axes(
    fig=cmap_fig, span=(0.05, 0.75, 0.9, 0.2), style=sebplt.AXES_MATPLOTLIB
)
cmap_mappable = atmospheric_cherenkov_response.plot.energy_cmap(
    energy_start_GeV=ENERGY_START_GEV, energy_stop_GeV=ENERGY_STOP_GEV
)
plt.colorbar(cmap_mappable, cax=cmap_ax, orientation="horizontal")
cmap_ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
cmap_fig.savefig(os.path.join(out_dir, "energy_colorbar.jpg"))
sebplt.close(cmap_fig)

energy_bin = binning_utils.Binning(bin_edges=np.geomspace(1e-1, 1e2, 4))
SITES, PARTICLES = mdfl.production.find_site_and_particle_keys(
    work_dir=work_dir
)
SITES = ["chile"]


grid_vertices = mdfl.allsky.hemisphere.make_vertices(
    num_vertices=6, max_zenith_distance_deg=70
)
grid_faces = mdfl.allsky.hemisphere.make_faces(vertices=grid_vertices)

grid_vertices_az_zd = []
for i in range(len(grid_vertices)):
    (az, zd) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
        cx=grid_vertices[i, 0],
        cy=grid_vertices[i, 1],
    )
    grid_vertices_az_zd.append([az, zd])
grid_vertices_az_zd = np.array(grid_vertices_az_zd)


finedome = mdfl.allsky.hemisphere.Grid(num_vertices=512)

res = {}
for sk in SITES:
    res[sk] = {}
    for pk in PARTICLES:
        res[sk][pk] = {}
        print("load", sk, pk)
        allsky = mdfl.allsky.AllSky(
            work_dir=os.path.join(work_dir, sk, pk),
            cache_dtype=mdfl.allsky.store.page.dtype(),
        )

        res[sk][pk]["grid"] = np.nan * np.ones(
            shape=(energy_bin["num"], len(grid_vertices_az_zd), 2)
        )
        for gbin, grid_az_zd in enumerate(grid_vertices_az_zd):
            print(gbin, "of", len(grid_vertices_az_zd))

            showers = mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
                allsky=allsky,
                azimuth_deg=grid_az_zd[0],
                zenith_deg=grid_az_zd[1],
                half_angle_deg=half_angle_deg,
                min_num_cherenkov_photons=1e3,
            )

            for ebin in range(energy_bin["num"]):
                _Estart = energy_bin["edges"][ebin]
                _Estop = energy_bin["edges"][ebin + 1]
                _E = showers["particle_energy_GeV"]
                mask = np.logical_and(_E >= _Estart, _E < _Estop)
                _cx = showers["particle_cx_rad"][mask]
                _cy = showers["particle_cy_rad"][mask]

                fd_hist = np.zeros(len(finedome.faces), dtype=int)
                for j in range(len(_cx)):
                    iface = finedome.query_cx_cy(cx=_cx[j], cy=_cy[j])
                    fd_hist[iface] += 1
                imax_face = np.argmax(fd_hist)
                max_face = finedome.faces[imax_face]
                aa = finedome.vertices[max_face[0]]
                bb = finedome.vertices[max_face[1]]
                cc = finedome.vertices[max_face[2]]
                _cx = np.mean([aa[0], bb[0], cc[0]])
                _cy = np.mean([aa[1], bb[1], cc[1]])

                (
                    _az_deg,
                    _zd_deg,
                ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
                    cx=_cx,
                    cy=_cy,
                )
                res[sk][pk]["grid"][ebin][gbin] = np.array([_az_deg, _zd_deg])
            del showers
        del allsky


FIELD_OF_VIEW = {
    "angle_deg": 90,
    "particles": PARTICLES,
    "zenith_mayor_deg": [0, 20, 40, 60, 80],
    "zenith_minor_deg": [0, 10, 20, 30, 40, 40, 50, 60, 70, 80, 90],
}

for skey in res:
    for pkey in res[skey]:
        deflgrid = res[skey][pkey]

        fov_deg = FIELD_OF_VIEW["angle_deg"]
        azimuth_mayor_deg = np.linspace(0, 360, 12, endpoint=False)
        zenith_mayor_deg = FIELD_OF_VIEW["zenith_mayor_deg"]
        azimuth_minor_deg = np.linspace(0, 360, 24, endpoint=False)
        zenith_minor_deg = FIELD_OF_VIEW["zenith_minor_deg"]
        rfov = np.sin(np.deg2rad(fov_deg))

        print(skey, pkey)

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
            linewidth=0.5,
            color=GRID_COLOR,
            alpha=1.0,
            draw_lower_horizontal_edge_deg=fov_deg,
        )
        # minor
        sebplt.hemisphere.ax_add_grid(
            ax=ax,
            azimuths_deg=azimuth_minor_deg,
            zeniths_deg=zenith_minor_deg,
            linewidth=0.5 * 0.5,
            color=GRID_COLOR,
            alpha=1.0,
            draw_lower_horizontal_edge_deg=None,
        )

        sebplt.hemisphere.ax_add_mesh(
            ax=ax,
            azimuths_deg=grid_vertices_az_zd[:, 0],
            zeniths_deg=grid_vertices_az_zd[:, 1],
            faces=grid_faces,
            color="black",
            linestyle="-",
            linewidth=0.1,
        )

        for ebin in range(energy_bin["num"]):
            rgbas = np.array(cmap_mappable.to_rgba(energy_bin["edges"][ebin]))
            rgbas[3] = ALPHA

            sebplt.hemisphere.ax_add_mesh(
                ax=ax,
                azimuths_deg=deflgrid["grid"][ebin, :, 0],
                zeniths_deg=deflgrid["grid"][ebin, :, 1],
                faces=grid_faces,
                color=rgbas,
                linestyle="-",
                linewidth=1,
            )

        ax.text(-1.0 * rfov, -1.0 * rfov, "{:1.1f}$^\\circ$".format(fov_deg))
        ax.set_axis_off()
        ax.set_aspect("equal")
        sebplt.hemisphere.ax_add_ticklabels(
            ax=ax, azimuths_deg=[0, 90, 180, 270], rfov=0.93 * rfov
        )
        ax.set_xlim([-1.01 * rfov, 1.01 * rfov])
        ax.set_ylim([-1.01 * rfov, 1.01 * rfov])
        fig.savefig(
            os.path.join(
                out_dir,
                "{:s}_{:s}.jpg".format(skey, pkey),
            )
        )
        sebplt.close(fig)
