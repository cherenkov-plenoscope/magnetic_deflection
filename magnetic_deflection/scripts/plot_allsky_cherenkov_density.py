import argparse
import magnetic_deflection as mdfl
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot


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
    default=6.5,
    help="Half angle of instrument's field-of-view.",
)

args = parser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])
FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}

NE = 129
ND = 65
energy_bin = binning_utils.Binning(bin_edges=np.geomspace(1e-1, 1e2, NE))

POINTING = {
    "azimuth_rad": np.deg2rad(args.azimuth_deg),
    "zenith_rad": np.deg2rad(args.zenith_deg),
    "half_angle_rad": np.deg2rad(args.half_angle_deg),
}

site_keys, particle_keys = mdfl.production.find_site_and_particle_keys(
    work_dir=work_dir
)

res = {}
for sk in site_keys:
    res[sk] = {}
    for pk in particle_keys:
        res[sk][pk] = {}

        print("load", sk, pk)

        allsky = mdfl.allsky.AllSky(
            work_dir=os.path.join(work_dir, sk, pk),
            cache_dtype=mdfl.allsky.store.page.dtype(),
        )

        showers = mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
            allsky=allsky,
            azimuth_rad=POINTING["azimuth_rad"],
            zenith_rad=POINTING["zenith_rad"],
            half_angle_rad=POINTING["half_angle_rad"],
            min_num_cherenkov_photons=1e3,
        )

        res[sk][pk]["particle_energy_GeV"] = showers["particle_energy_GeV"]
        res[sk][pk]["cherenkov_num_photons"] = showers["cherenkov_num_photons"]
        res[sk][pk]["cherenkov_maximum_asl_m"] = showers[
            "cherenkov_maximum_asl_m"
        ]
        for percentile in [50, 90]:
            dens = mdfl.allsky.analysis.estimate_cherenkov_density(
                showers=showers,
                percentile=percentile,
            )
            res[sk][pk].update(dens)

        del showers
        del allsky
        del dens


DENSITIES = {
    "cherenkov_area_density50_per_m2": {
        "label": "(50 percentile)\narea density",
        "unit": r"m$^{-2}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-5, 1e2, ND)),
    },
    "cherenkov_area_density90_per_m2": {
        "label": "(90 percentile) area density",
        "unit": r"m$^{-2}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-5, 1e2, ND)),
    },
    "cherenkov_light_field_density50_per_m2_per_sr": {
        "label": "(50 percentile)\nlight-field density",
        "unit": r"m$^{-2}$ sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-5, 1e5, ND)),
    },
    "cherenkov_light_field_density90_per_m2_per_sr": {
        "label": "(90 percentile)\nlight-field density",
        "unit": r"m$^{-2}$ sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-5, 1e5, ND)),
    },
    "cherenkov_solid_angle_density50_per_sr": {
        "label": "(50 percentile)\nsolid angle density",
        "unit": r"sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e9, ND)),
    },
    "cherenkov_solid_angle_density90_per_sr": {
        "label": "(90 percentile)\nsolid angle density",
        "unit": r"sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e9, ND)),
    },
    "cherenkov_num_photons": {
        "label": "size",
        "unit": r"1",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e7, ND)),
    },
    "cherenkov_area50_m2": {
        "label": "(50 percentile)\narea",
        "unit": r"m$^2$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e4, 1e12, ND)),
    },
    "cherenkov_area90_m2": {
        "label": "(90 percentile)\narea",
        "unit": r"m$^2$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e4, 1e12, ND)),
    },
    "cherenkov_solid_angle50_sr": {
        "label": "(50 percentile)\nsolid angle",
        "unit": r"sr",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-4, 1e1, ND)),
    },
    "cherenkov_solid_angle90_sr": {
        "label": "(90 percentile)\nsolid angle",
        "unit": r"sr",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-4, 1e1, ND)),
    },
    "cherenkov_maximum_asl_m": {
        "label": "altitude of maximum",
        "unit": r"m",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(5e3, 3e4, ND)),
        "log": False,
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
        if len(ysplit[ix]) > 0:
            y_low[ix] = np.percentile(a=ysplit[ix], q=1e2 * qlow)
            y_center[ix] = np.percentile(a=ysplit[ix], q=1e2 * qcenter)
            y_high[ix] = np.percentile(a=ysplit[ix], q=1e2 * qhigh)

    return y_low, y_center, y_high


# histogram style
# ---------------
for dkey in DENSITIES:
    for sk in res:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=(0.175, 0.2, 0.8, 0.75))

        for pk in particle_keys:
            dhist = np.histogram2d(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][dkey],
                bins=(energy_bin["edges"], DENSITIES[dkey]["bin"]["edges"]),
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
                DENSITIES[dkey]["bin"]["edges"],
                qhist.T,
                cmap=PLT["particles"][pk]["cmap"],
            )

        if "log" in DENSITIES[dkey]:
            if DENSITIES[dkey]["log"]:
                ax.semilogy()
        else:
            ax.semilogy()
        ax.semilogx()
        ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
        ax.set_ylabel(
            DENSITIES[dkey]["label"]
            + PLT["label_unit_seperator"]
            + DENSITIES[dkey]["unit"]
        )
        ax.set_ylim(DENSITIES[dkey]["bin"]["limits"])
        ax.set_xlim(energy_bin["limits"])
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(sk, dkey)))
        sebplt.close(fig)


# percentile style
# ----------------
for dkey in DENSITIES:
    for sk in res:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=(0.175, 0.2, 0.8, 0.75))

        for pk in particle_keys:
            ylow, ycen, yhig = estimate_median_and_percentiles(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][dkey],
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

        if "log" in DENSITIES[dkey]:
            if DENSITIES[dkey]["log"]:
                ax.semilogy()
        else:
            ax.semilogy()
        ax.semilogx()
        ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
        ax.set_ylabel(
            DENSITIES[dkey]["label"]
            + PLT["label_unit_seperator"]
            + DENSITIES[dkey]["unit"]
        )
        ax.set_ylim(DENSITIES[dkey]["bin"]["limits"])
        ax.set_xlim(energy_bin["limits"])
        fig.savefig(
            os.path.join(out_dir, "{:s}_{:s}_qstyle.jpg".format(sk, dkey))
        )
        sebplt.close(fig)
