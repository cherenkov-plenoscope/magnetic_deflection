import argparse
import os
import numpy as np
import json_utils
import magnetic_deflection as mdfl
import binning_utils
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import sklearn
from sklearn import cluster


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
HEMISPHERE_AXSTYLE = {"spines": [], "axes": [], "grid": False}

PROBABLY_BEYOND_THE_HORIZON_ANGLE_DEG = 60
GRID_COLOR = (0.5, 0.5, 0.5)
MARKER_HALF_ANGLE_DEG = 1.25
ALPHA = 0.75

# energy colorbar
# ---------------
EE = mdfl.common_settings_for_plotting.common_energy_limits()

cmap_fig = sebplt.figure(CMAP_FIGSIZE)
cmap_ax = sebplt.add_axes(
    fig=cmap_fig, span=(0.05, 0.75, 0.9, 0.2), style=sebplt.AXES_MATPLOTLIB
)
cmap_mappable = atmospheric_cherenkov_response.plot.energy_cmap(**EE)

plt.colorbar(cmap_mappable, cax=cmap_ax, orientation="horizontal")
cmap_ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
cmap_fig.savefig(os.path.join(out_dir, "energy_colorbar.jpg"))
sebplt.close(cmap_fig)

energy_bin = binning_utils.Binning(
    bin_edges=np.geomspace(EE["energy_start_GeV"], EE["energy_stop_GeV"], 9)
)
SITES, PARTICLES = mdfl.production.find_site_and_particle_keys(
    work_dir=work_dir
)

sample_directions = binning_utils.sphere.fibonacci_space(
    size=24,
    max_zenith_distance_rad=np.deg2rad(50),
)


def cxcycz_2_az_zd_deg(cxcycz):
    out_az_zd = np.zeros(shape=(len(cxcycz), 2))
    for i in range(len(cxcycz)):
        (az, zd) = mdfl.spherical_coordinates._cx_cy_cz_to_az_zd_deg(
            cx=cxcycz[i, 0],
            cy=cxcycz[i, 1],
            cz=cxcycz[i, 2],
        )
        out_az_zd[i] = np.array([az, zd])
    return out_az_zd


sample_directions_az_zd = cxcycz_2_az_zd_deg(sample_directions)


res = {}
for sk in SITES:
    res[sk] = {}
    for pk in PARTICLES:
        res[sk][pk] = {}
        print("load", sk, pk)

        cache_path = os.path.join(out_dir, "{:s}_{:s}.json".format(sk, pk))
        if os.path.exists(cache_path):
            with open(cache_path, "rt") as f:
                res[sk][pk]["grid"] = json_utils.loads(f.read())
        else:
            res[sk][pk]["grid"] = np.nan * np.ones(
                shape=(energy_bin["num"], len(sample_directions_az_zd), 2)
            )

            allsky = mdfl.allsky.AllSky(
                work_dir=os.path.join(work_dir, sk, pk),
                cache_dtype=mdfl.allsky.store.page.dtype(),
            )

            for gbin, grid_az_zd in enumerate(sample_directions_az_zd):
                showers = (
                    mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
                        allsky=allsky,
                        azimuth_deg=grid_az_zd[0],
                        zenith_deg=grid_az_zd[1],
                        half_angle_deg=half_angle_deg,
                        min_num_cherenkov_photons=1e3,
                    )
                )

                for ebin in range(energy_bin["num"]):
                    print(
                        gbin, "of", len(sample_directions_az_zd), "ebin", ebin
                    )

                    _Estart = energy_bin["edges"][ebin]
                    _Estop = energy_bin["edges"][ebin + 1]
                    _E = showers["particle_energy_GeV"]
                    mask = np.logical_and(_E >= _Estart, _E < _Estop)
                    _cx = showers["particle_cx_rad"][mask]
                    _cy = showers["particle_cy_rad"][mask]

                    if len(_cx) > 0:
                        _cz = mdfl.spherical_coordinates.restore_cz(
                            cx=_cx, cy=_cy
                        )
                        X = np.c_[_cx, _cy, _cz]
                        print("start MS", len(_cx))
                        min_bin_freq = max([1, int(0.01 * len(_cx))])
                        ms = sklearn.cluster.MeanShift(
                            bandwidth=np.deg2rad(10),
                            bin_seeding=True,
                            min_bin_freq=min_bin_freq,
                        )
                        print("start MS.fit")
                        ms.fit(X=X)
                        print("done MS.fit")
                        max_density_cxcycz = ms.cluster_centers_[0]
                        (
                            max_density_az_deg,
                            max_density_zd_deg,
                        ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
                            cx=max_density_cxcycz[0],
                            cy=max_density_cxcycz[1],
                        )
                        res[sk][pk]["grid"][ebin][gbin] = np.array(
                            [max_density_az_deg, max_density_zd_deg]
                        )

            with open(cache_path, "wt") as f:
                f.write(json_utils.dumps(res[sk][pk]["grid"]))

            del allsky


FIELD_OF_VIEW = mdfl.common_settings_for_plotting.hemisphere_field_of_view()[
    "wide"
]


def plane_normal(az1_deg, zd1_deg, az2_deg, zd2_deg):
    az_zd_to_cx_cy_cz = mdfl.spherical_coordinates._az_zd_to_cx_cy_cz
    c1 = np.array(az_zd_to_cx_cy_cz(az1_deg, zd1_deg))
    c2 = np.array(az_zd_to_cx_cy_cz(az2_deg, zd2_deg))
    assert 0.95 < np.linalg.norm(c1) < 1.05
    assert 0.95 < np.linalg.norm(c2) < 1.05
    n = np.cross(c1, c2)
    return n / np.linalg.norm(n)


for sk in res:
    sss = atmospheric_cherenkov_response.sites.init(sk)
    mag = mdfl.common_settings_for_plotting.magnetic_flux(
        earth_magnetic_field_x_muT=sss["earth_magnetic_field_x_muT"],
        earth_magnetic_field_z_muT=sss["earth_magnetic_field_z_muT"],
    )

    for pk in res[sk]:
        deflgrid = res[sk][pk]

        azimuth_minor_deg = FIELD_OF_VIEW["azimuth_minor_deg"]
        zenith_minor_deg = FIELD_OF_VIEW["zenith_minor_deg"]
        rfov = FIELD_OF_VIEW["rfov"]
        print(sk, pk)

        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(
            fig=fig,
            span=(0.02, 0.02, 0.96, 0.96),
            style=HEMISPHERE_AXSTYLE,
        )

        sebplt.hemisphere.ax_add_grid(
            ax=ax,
            azimuths_deg=azimuth_minor_deg,
            zeniths_deg=zenith_minor_deg,
            linewidth=0.05,
            color=GRID_COLOR,
            alpha=1.0,
            draw_lower_horizontal_edge_deg=None,
            zenith_min_deg=5,
        )

        if mag["magnitude_uT"] > 1e-6:
            sebplt.hemisphere.ax_add_magnet_flux_symbol(
                ax=ax,
                azimuth_deg=mag["azimuth_deg"],
                zenith_deg=mag["zenith_deg"],
                half_angle_deg=2.5,
                color="black",
                direction="inwards" if mag["sign"] > 0 else "outwards",
            )

        for gbin in range(len(sample_directions)):
            for ebin in range(energy_bin["num"]):
                rgb = cmap_mappable.to_rgba(energy_bin["centers"][ebin])
                sebplt.hemisphere.ax_add_projected_circle(
                    ax=ax,
                    azimuth_deg=deflgrid["grid"][ebin, gbin, 0],
                    zenith_deg=deflgrid["grid"][ebin, gbin, 1],
                    half_angle_deg=MARKER_HALF_ANGLE_DEG,
                    fill=True,
                    facecolor=rgb,
                    linewidth=0.0,
                    alpha=ALPHA,
                    zorder=2,
                )

                if ebin > 0:
                    _start_ebin = ebin - 1
                    _stop_ebin = ebin

                    _start = deflgrid["grid"][_start_ebin, gbin, :]
                    _stop = deflgrid["grid"][_stop_ebin, gbin, :]
                    _line = mdfl.spherical_coordinates.make_great_circle_line(
                        start_azimuth_deg=_start[0],
                        start_zenith_deg=_start[1],
                        stop_azimuth_deg=_stop[0],
                        stop_zenith_deg=_stop[1],
                    )

                    _start_color = cmap_mappable.to_rgba(
                        energy_bin["centers"][_start_ebin]
                    )
                    _stop_color = cmap_mappable.to_rgba(
                        energy_bin["centers"][_stop_ebin]
                    )
                    _linestyle = "-"
                    _linewidth = 1.0
                    _linecolor = np.mean(
                        [_start_color, _stop_color],
                        axis=0,
                    )

                    if ebin < energy_bin["num"] - 1:
                        _next = deflgrid["grid"][ebin + 1, gbin, :]

                        _next_valid = not np.any(np.isnan(_next))
                        _start_valid = not np.any(np.isnan(_start))
                        _stop_valid = not np.any(np.isnan(_stop))

                        if _next_valid and _start_valid and _stop_valid:
                            _n = plane_normal(
                                _start[0], _start[1], _stop[0], _stop[1]
                            )
                            _l = plane_normal(
                                _stop[0], _stop[1], _next[0], _next[1]
                            )
                            _delta_rad = mdfl.spherical_coordinates._angle_between_vectors_rad(
                                a=_n, b=_l
                            )
                            _delta_deg = np.rad2deg(_delta_rad)

                            if (
                                _delta_deg
                                > PROBABLY_BEYOND_THE_HORIZON_ANGLE_DEG
                            ):
                                _linestyle = ":"
                                _linewidth = 0.5
                                _linecolor = "gray"

                    sebplt.hemisphere.ax_add_plot(
                        ax=ax,
                        azimuths_deg=_line[0],
                        zeniths_deg=_line[1],
                        color=_linecolor,
                        linewidth=_linewidth,
                        linestyle=_linestyle,
                    )

        ax.set_axis_off()
        ax.set_aspect("equal")
        sebplt.hemisphere.ax_add_ticklabel_text(
            ax=ax,
            radius=0.95 * rfov,
            label_azimuths_deg=[0, 90, 180, 270],
            label_azimuths=["N", "E", "S", "W"],
            xshift=-0.05,
            yshift=-0.025,
            fontsize=8,
        )
        ax.set_xlim([-1.01 * rfov, 1.01 * rfov])
        ax.set_ylim([-1.01 * rfov, 1.01 * rfov])
        fig.savefig(
            os.path.join(
                out_dir,
                "{:s}_{:s}.jpg".format(sk, pk),
            )
        )
        sebplt.close(fig)
