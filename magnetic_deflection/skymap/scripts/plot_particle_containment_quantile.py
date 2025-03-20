import argparse
import os
import numpy as np
import corsika_primary
import magnetic_deflection as mdfl
import solid_angle_utils
import binning_utils
import spherical_coordinates
import plenoirf
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import json_utils


parser = argparse.ArgumentParser(
    prog="plot_particle_containment_quantile.py",
    description=(
        "Show the containment quantile of particles vs. " "the solid angle."
    ),
)
parser.add_argument(
    "--skymap_dir",
    metavar="PATH",
    type=str,
    help="Skymap working directory.",
)
parser.add_argument(
    "--out_dir",
    metavar="PATH",
    type=str,
    help="Directory to write figures to.",
)
parser.add_argument(
    "--max_zenith_deg",
    metavar="FLOAT",
    type=float,
    default=45.0,
    help="Maximum zenith distance pointing of instrument.",
)
parser.add_argument(
    "--half_angle_deg",
    metavar="FLOAT",
    type=float,
    default=3.25,
    help="Half angle of instrument's field-of-view.",
)

args = parser.parse_args()

skymap = mdfl.skymap.SkyMap(args.skymap_dir)
out_dir = args.out_dir
instrument_pointing_max_zenith_rad = np.deg2rad(args.max_zenith_deg)
instrument_field_of_view_half_angle_rad = np.deg2rad(args.half_angle_deg)
os.makedirs(out_dir, exist_ok=True)
prng = np.random.Generator(np.random.PCG64(seed=18))

sk = skymap.config["site"]["key"]
pk = skymap.config["particle"]["key"]

energy_bin = binning_utils.Binning(bin_edges=skymap.binning["energy"]["edges"])
NUM_POINTINGS = energy_bin["num"]

NUM_SOLID_ANGLE_BINS = 128
_solid_angle_sr = 0.32

sky_faces_solid_angles_sr = skymap.binning["sky"].faces_solid_angles

common_solid_angle_bin = binning_utils.Binning(
    bin_edges=np.geomspace(
        start=np.median(skymap.binning["sky"].faces_solid_angles),
        stop=np.sum(skymap.binning["sky"].faces_solid_angles),
        num=NUM_SOLID_ANGLE_BINS + 1,
    ),
    weight_lower_edge=0.5,
)

quantile_in_common_solid_angle = np.zeros(
    (energy_bin["num"], NUM_POINTINGS, common_solid_angle_bin["num"])
)

for eee in range(energy_bin["num"]):
    energy_start_GeV = skymap.binning["energy"]["edges"][eee]
    energy_stop_GeV = skymap.binning["energy"]["edges"][eee + 1]

    for ppp in range(NUM_POINTINGS):
        (instrument_pointing_azimuth_rad, instrument_pointing_zenith_rad) = (
            spherical_coordinates.random.uniform_az_zd_in_cone(
                prng=prng,
                azimuth_rad=0.0,
                zenith_rad=0.0,
                min_half_angle_rad=0.0,
                max_half_angle_rad=instrument_pointing_max_zenith_rad,
                size=1,
            )
        )

        result, debug = skymap.draw(
            azimuth_rad=instrument_pointing_azimuth_rad[0],
            zenith_rad=instrument_pointing_zenith_rad[0],
            half_angle_rad=instrument_field_of_view_half_angle_rad,
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
            threshold_cherenkov_density_per_sr=0.0,
            solid_angle_sr=_solid_angle_sr,
            prng=prng,
        )

        (
            quantile,
            quantile_solid_angle_sr,
        ) = mdfl.skymap.estimate_sky_cherenkov_quantile_vs_solid_angle(
            sky_cherenkov_per_sr=debug["sky_cherenkov_per_sr"],
            sky_faces_solid_angles_sr=sky_faces_solid_angles_sr,
        )

        _quantile_in_common_solid_angle = np.interp(
            x=common_solid_angle_bin["centers"],
            xp=quantile_solid_angle_sr,
            fp=quantile,
        )

        quantile_in_common_solid_angle[eee, ppp] = (
            _quantile_in_common_solid_angle
        )

rrr = {}
rrr["p16"] = np.zeros((energy_bin["num"], common_solid_angle_bin["num"]))
rrr["p50"] = np.zeros((energy_bin["num"], common_solid_angle_bin["num"]))
rrr["p84"] = np.zeros((energy_bin["num"], common_solid_angle_bin["num"]))

for eee in range(energy_bin["num"]):
    rrr["p16"][eee] = np.percentile(
        quantile_in_common_solid_angle[eee], 16, axis=0
    )
    rrr["p50"][eee] = np.percentile(
        quantile_in_common_solid_angle[eee], 50, axis=0
    )
    rrr["p84"][eee] = np.percentile(
        quantile_in_common_solid_angle[eee], 84, axis=0
    )


cmap = plenoirf.summary.figure.make_particle_colormaps(
    particle_colors=plenoirf.summary.figure.PARTICLE_COLORS
)

PLOT_METHOD = {"pcolormesh": 1, "contour": 2}

for plot_method in PLOT_METHOD:
    fig = sebplt.figure({"rows": 720, "cols": 1280, "fontsize": 1})
    ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])
    sebplt.ax_add_box(
        ax=ax_c,
        xlim=energy_bin["limits"],
        ylim=common_solid_angle_bin["limits"],
        color="grey",
        linewidth=0.5,
        linestyle="--",
    )
    if plot_method == "contour":
        _pcm_confusion = ax_c.contour(
            energy_bin["centers"],
            common_solid_angle_bin["centers"],
            np.transpose(rrr["p50"]),
            cmap=cmap[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        ax_c.clabel(_pcm_confusion, fontsize=5)
    elif plot_method == "pcolormesh":
        _pcm_confusion = ax_c.pcolormesh(
            energy_bin["centers"],
            common_solid_angle_bin["centers"],
            np.transpose(rrr["p50"]),
            cmap=cmap[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
    ax_c.set_xlim([0.1, 100])
    ax_c.set_ylim([1e-3, 2 * np.pi])
    ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_c.set_xlabel("energy / GeV")
    ax_c.set_ylabel("solid angle / sr")
    ax_c.loglog()

    fig.savefig(
        os.path.join(
            out_dir,
            f"{sk:s}_{pk:s}_containment_vs_solid_angle_{plot_method:s}.jpg",
        )
    )
    sebplt.close(fig)
