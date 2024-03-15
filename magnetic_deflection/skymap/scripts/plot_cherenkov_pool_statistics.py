import argparse
import magnetic_deflection as mdfl
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import numpy as np


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
    "--azimuth_deg",
    metavar="FLOAT",
    type=float,
    default=0.0,
    help="Azimuth pointing of instrument.",
)
parser.add_argument(
    "--zenith_deg",
    metavar="FLOAT",
    type=float,
    default=0.0,
    help="Zenith distance pointing of instrument.",
)
parser.add_argument(
    "--half_angle_deg",
    metavar="FLOAT",
    type=float,
    default=15,
    help="Half angle of instrument's field-of-view.",
)

args = parser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])
FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}
EE = mdfl.common_settings_for_plotting.common_energy_limits()


NE = 129
ND = 65
energy_bin = binning_utils.Binning(bin_edges=np.geomspace(1e-1, 1e2, NE))

POINTING = {
    "azimuth_rad": np.deg2rad(args.azimuth_deg),
    "zenith_rad": np.deg2rad(args.zenith_deg),
    "half_angle_rad": np.deg2rad(args.half_angle_deg),
}

site_keys, particle_keys = mdfl.find_site_and_particle_keys(work_dir=work_dir)

mask_function = mdfl.cherenkov_pool.reports.MaskPrimaryInCone(
    azimuth_rad=POINTING["azimuth_rad"],
    zenith_rad=POINTING["zenith_rad"],
    half_angle_rad=POINTING["half_angle_rad"],
)

res = {}
for sk in site_keys:
    res[sk] = {}
    for pk in particle_keys:
        res[sk][pk] = {}

        print("load", sk, pk)

        reports = mdfl.cherenkov_pool.reports.read(
            path=os.path.join(work_dir, sk, pk, "results", "reports.tar"),
            mask_function=mask_function,
        )

        for name in reports.dtype.names:
            res[sk][pk][name] = reports[name]

            nkey = "cherenkov_size_per_area_p??_per_m2"

            for pp in [16, 50, 84]:
                quantile = pp / 100
                res[sk][pk][
                    "cherenkov_size_per_area_p{:02d}_per_m2".format(pp)
                ] = (
                    quantile
                    * reports["cherenkov_num_photons"]
                    / reports[
                        "cherenkov_containment_area_p{:02d}_m2".format(pp)
                    ]
                )
                res[sk][pk][
                    "cherenkov_size_per_solid_angle_p{:02d}_per_sr".format(pp)
                ] = (
                    quantile
                    * reports["cherenkov_num_photons"]
                    / reports[
                        "cherenkov_containment_solid_angle_p{:02d}_sr".format(
                            pp
                        )
                    ]
                )

                res[sk][pk]["cherenkov_etendue_p{:02d}_m2_sr".format(pp)] = (
                    reports[
                        "cherenkov_containment_solid_angle_p{:02d}_sr".format(
                            pp
                        )
                    ]
                    * reports[
                        "cherenkov_containment_area_p{:02d}_m2".format(pp)
                    ]
                )

                res[sk][pk][
                    "cherenkov_size_per_etendue_p{:02d}_per_m2_per_sr".format(
                        pp
                    )
                ] = (
                    quantile
                    * reports["cherenkov_num_photons"]
                    / res[sk][pk]["cherenkov_etendue_p{:02d}_m2_sr".format(pp)]
                )

DENSITIES = {
    "cherenkov_altitude_p{:02d}_m": {
        "label": "altitude of maximum",
        "unit": r"m",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(5e3, 3e4, ND)),
        "log": False,
    },
    "cherenkov_containment_area_p{:02d}_m2": {
        "label": "area",
        "unit": r"m$^2$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e4, 1e7, ND)),
    },
    "cherenkov_containment_solid_angle_p{:02d}_sr": {
        "label": "solid angle",
        "unit": r"sr",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-3, 1e0, ND)),
    },
    "cherenkov_num_photons": {
        "label": "size",
        "unit": r"1",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e7, ND)),
    },
    "cherenkov_size_per_area_p{:02d}_per_m2": {
        "label": "size per area",
        "unit": r"m$^{-2}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-4, 1e2, ND)),
    },
    "cherenkov_size_per_solid_angle_p{:02d}_per_sr": {
        "label": "size per solid angle",
        "unit": r"sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e9, ND)),
    },
    "cherenkov_size_per_etendue_p{:02d}_per_m2_per_sr": {
        "label": "size per etendue",
        "unit": r"m$^{-2}$ sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-2, 1e4, ND)),
    },
}


def estimate_median_and_percentiles(
    x, y, xbin, qlow=0.5 - 0.68 / 2, qcenter=0.5, qhigh=0.5 + 0.68 / 2
):
    ysplit = []
    for ix in range(xbin["num"]):
        xstart = xbin["edges"][ix]
        xstop = xbin["edges"][ix + 1]
        mask = np.logical_and(x >= xstart, x < xstop)
        ysplit.append(y[mask])

    y_center = np.nan * np.ones(xbin["num"])
    y_low = np.nan * np.ones(xbin["num"])
    y_high = np.nan * np.ones(xbin["num"])
    for ix in range(xbin["num"]):
        valid = np.logical_not(np.isnan(ysplit[ix]))
        if np.sum(valid) > 0.1 * len(valid):
            y_low[ix] = np.percentile(a=ysplit[ix][valid], q=1e2 * qlow)
            y_center[ix] = np.percentile(a=ysplit[ix][valid], q=1e2 * qcenter)
            y_high[ix] = np.percentile(a=ysplit[ix][valid], q=1e2 * qhigh)

    return y_low, y_center, y_high


# histogram style
# ---------------
for _dkey in DENSITIES:
    dkey = _dkey.format(50)
    for sk in res:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=(0.175, 0.2, 0.8, 0.75))

        for pk in particle_keys:
            dhist = np.histogram2d(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][dkey],
                bins=(energy_bin["edges"], DENSITIES[_dkey]["bin"]["edges"]),
            )[0]
            ehist = np.histogram(
                res[sk][pk]["particle_energy_GeV"],
                bins=energy_bin["edges"],
            )[0]

            qhist = np.zeros(shape=dhist.shape)
            for ebin in range(energy_bin["num"]):
                if np.sum(ehist[ebin]) > 0:
                    qhist[ebin, :] = dhist[ebin, :] / ehist[ebin]

            ax.pcolormesh(
                energy_bin["edges"],
                DENSITIES[_dkey]["bin"]["edges"],
                qhist.T,
                cmap=PLT["particles"][pk]["cmap"],
            )

        if "log" in DENSITIES[_dkey]:
            if DENSITIES[_dkey]["log"]:
                ax.semilogy()
        else:
            ax.semilogy()
        ax.semilogx()
        ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
        ax.set_ylabel(
            DENSITIES[_dkey]["label"]
            + PLT["label_unit_seperator"]
            + DENSITIES[_dkey]["unit"]
        )
        ax.set_ylim(DENSITIES[_dkey]["bin"]["limits"])
        ax.set_xlim(energy_bin["limits"])
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(sk, dkey)))
        sebplt.close(fig)


# percentile style
# ----------------
for _dkey in DENSITIES:
    dkey = _dkey.format(50)
    for sk in res:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=(0.175, 0.2, 0.8, 0.75))

        for pk in particle_keys:
            _, yhig, _ = estimate_median_and_percentiles(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][_dkey.format(84)],
                xbin=energy_bin,
            )
            _, ycen, _ = estimate_median_and_percentiles(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][_dkey.format(50)],
                xbin=energy_bin,
            )
            _, ylow, _ = estimate_median_and_percentiles(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][_dkey.format(16)],
                xbin=energy_bin,
            )

            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=ycen,
                linestyle="-",
                linecolor=PLT["particles"][pk]["color"],
                linealpha=1.0,
                bincounts_upper=yhig,
                bincounts_lower=ylow,
                face_color=PLT["particles"][pk]["color"],
                face_alpha=0.25,
                label=None,
                draw_bin_walls=False,
            )

        if "log" in DENSITIES[_dkey]:
            if DENSITIES[_dkey]["log"]:
                ax.semilogy()
        else:
            ax.semilogy()
        ax.semilogx()
        ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
        ax.set_ylabel(
            DENSITIES[_dkey]["label"]
            + PLT["label_unit_seperator"]
            + DENSITIES[_dkey]["unit"]
        )
        ax.set_ylim(DENSITIES[_dkey]["bin"]["limits"])
        ax.set_xlim(energy_bin["limits"])
        fig.savefig(
            os.path.join(out_dir, "{:s}_{:s}_qstyle.jpg".format(sk, dkey))
        )
        sebplt.close(fig)
