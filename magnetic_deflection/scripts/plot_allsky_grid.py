import argparse
import os
import numpy as np
import json_utils
import magnetic_deflection as mdfl
import spherical_coordinates
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
half_angle_rad = np.deg2rad(args.half_angle_deg)

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 2}
CMAP_FIGSIZE = {"rows": 300, "cols": 1280, "fontsize": 1.75}
HEMISPHERE_AXSTYLE = {"spines": [], "axes": [], "grid": False}

PROBABLY_BEYOND_THE_HORIZON_ANGLE_RAD = np.deg2rad(60)
GRID_COLOR = (0.5, 0.5, 0.5)
MARKER_HALF_ANGLE_RAD = np.deg2rad(1.25)
ALPHA = 0.75
CLUSTER_BANDWIDTH_RAD = np.deg2rad(10.0)
NUM_ENERGY_BINS = 8

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
    bin_edges=np.geomspace(
        EE["energy_start_GeV"],
        EE["energy_stop_GeV"],
        NUM_ENERGY_BINS + 1,
    )
)
SITES, PARTICLES = mdfl.find_site_and_particle_keys(work_dir=work_dir)
SITES, PARTICLES = (["chile"], ["electron", "proton"])

samples = {}
samples["cxcycz"] = binning_utils.sphere.fibonacci_space(
    size=24,
    max_zenith_distance_rad=np.deg2rad(50),
)
samples["num"] = len(samples["cxcycz"])
(
    samples["azimuth_rad"],
    samples["zenith_rad"],
) = spherical_coordinates.cx_cy_cz_to_az_zd(
    cx=samples["cxcycz"][:, 0],
    cy=samples["cxcycz"][:, 1],
    cz=samples["cxcycz"][:, 2],
)


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
                shape=(energy_bin["num"], samples["num"], 2)
            )

            allsky = mdfl.allsky.AllSky(
                work_dir=os.path.join(work_dir, sk, pk),
                cache_dtype=mdfl.allsky.store.page.dtype(),
            )

            for gbin in range(samples["num"]):
                showers = (
                    mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
                        allsky=allsky,
                        azimuth_rad=samples["azimuth_rad"][gbin],
                        zenith_rad=samples["zenith_rad"][gbin],
                        half_angle_rad=half_angle_rad,
                        min_num_cherenkov_photons=1e3,
                    )
                )

                for ebin in range(energy_bin["num"]):
                    print(
                        gbin + 1,
                        "/",
                        samples["num"],
                        " ",
                        ebin + 1,
                        "/",
                        energy_bin["num"],
                    )

                    _Estart = energy_bin["edges"][ebin]
                    _Estop = energy_bin["edges"][ebin + 1]
                    _E = showers["particle_energy_GeV"]
                    mask = np.logical_and(_E >= _Estart, _E < _Estop)
                    _cx = showers["particle_cx_rad"][mask]
                    _cy = showers["particle_cy_rad"][mask]

                    if len(_cx) > 0:
                        _cz = spherical_coordinates.restore_cz(cx=_cx, cy=_cy)
                        X = np.c_[_cx, _cy, _cz]
                        min_bin_freq = max([1, int(0.01 * len(_cx))])
                        ms = sklearn.cluster.MeanShift(
                            bandwidth=CLUSTER_BANDWIDTH_RAD,
                            bin_seeding=True,
                            min_bin_freq=min_bin_freq,
                        )
                        print("MeanShift.fit", len(_cx), "num showers")
                        ms.fit(X=X)
                        max_density_cxcycz = ms.cluster_centers_[0]
                        (
                            max_density_az_rad,
                            max_density_zd_rad,
                        ) = spherical_coordinates.cx_cy_to_az_zd(
                            cx=max_density_cxcycz[0],
                            cy=max_density_cxcycz[1],
                        )
                        res[sk][pk]["grid"][ebin][gbin] = np.array(
                            [max_density_az_rad, max_density_zd_rad]
                        )

            with open(cache_path, "wt") as f:
                f.write(json_utils.dumps(res[sk][pk]["grid"]))

            del allsky


def plane_normal(az1_rad, zd1_rad, az2_rad, zd2_rad):
    az_zd_to_cx_cy_cz = spherical_coordinates.az_zd_to_cx_cy_cz
    c1 = np.array(az_zd_to_cx_cy_cz(az1_rad, zd1_rad))
    c2 = np.array(az_zd_to_cx_cy_cz(az2_rad, zd2_rad))
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

        rfov = 1.0
        print(sk, pk)

        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(
            fig=fig,
            span=(0.02, 0.02, 0.96, 0.96),
            style=HEMISPHERE_AXSTYLE,
        )
        sebplt.hemisphere.ax_add_grid_stellarium_style(ax=ax)

        if mag["magnitude_uT"] > 1e-6:
            sebplt.hemisphere.ax_add_magnet_flux_symbol(
                ax=ax,
                azimuth_rad=mag["azimuth_rad"],
                zenith_rad=mag["zenith_rad"],
                half_angle_rad=np.deg2rad(2.5),
                color="black",
                direction="inwards" if mag["sign"] > 0 else "outwards",
            )

        for gbin in range(samples["num"]):
            for ebin in range(energy_bin["num"]):
                rgb = cmap_mappable.to_rgba(energy_bin["centers"][ebin])
                sebplt.hemisphere.ax_add_projected_circle(
                    ax=ax,
                    azimuth_rad=deflgrid["grid"][ebin, gbin, 0],
                    zenith_rad=deflgrid["grid"][ebin, gbin, 1],
                    half_angle_rad=MARKER_HALF_ANGLE_RAD,
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
                    (
                        _line_az,
                        _line_zd,
                    ) = mdfl.common_settings_for_plotting.make_great_circle_line(
                        start_azimuth_rad=_start[0],
                        start_zenith_rad=_start[1],
                        stop_azimuth_rad=_stop[0],
                        stop_zenith_rad=_stop[1],
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
                            _delta_rad = (
                                spherical_coordinates.angle_between_xyz(_n, _l)
                            )

                            if (
                                _delta_rad
                                > PROBABLY_BEYOND_THE_HORIZON_ANGLE_RAD
                            ):
                                _linestyle = ":"
                                _linewidth = 0.5
                                _linecolor = "gray"

                    sebplt.hemisphere.ax_add_plot(
                        ax=ax,
                        azimuths_rad=_line_az,
                        zeniths_rad=_line_zd,
                        color=_linecolor,
                        linewidth=_linewidth,
                        linestyle=_linestyle,
                    )

        ax.set_axis_off()
        ax.set_aspect("equal")
        sebplt.hemisphere.ax_add_ticklabel_text(
            ax=ax,
            radius=0.95 * rfov,
            label_azimuths_rad=[
                0,
                1 / 2 * np.pi,
                2 / 2 * np.pi,
                3 / 2 * np.pi,
            ],
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
