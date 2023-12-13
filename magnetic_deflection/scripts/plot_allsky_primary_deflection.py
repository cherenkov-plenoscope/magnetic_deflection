import argparse
import os
import numpy as np
import magnetic_deflection as mdfl
import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import json_utils


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
    default=2.5,
    help="Half angle of instrument's field-of-view.",
)

args = parser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

POINTING = {
    "azimuth_deg": args.azimuth_deg,
    "zenith_deg": args.zenith_deg,
    "half_angle_deg": args.half_angle_deg,
}

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 2}
CMAP_FIGSIZE = {"rows": 300, "cols": 1280, "fontsize": 1.75}

ON_AXIS_SCALE = 1.0

HEMISPHERE_AXSTYLE = {"spines": [], "axes": [], "grid": False}

EE = mdfl.examples.common_energy_limits()
GRID_COLOR = (0.5, 0.5, 0.5)
FRACTION = 0.1  # 1.0
ALPHA = 0.1

# energy colorbar
# ---------------

cmap_fig = sebplt.figure(CMAP_FIGSIZE)
cmap_ax = sebplt.add_axes(
    fig=cmap_fig, span=(0.05, 0.75, 0.9, 0.2), style=sebplt.AXES_MATPLOTLIB
)
cmap_mappable = atmospheric_cherenkov_response.plot.energy_cmap(**EE)
plt.colorbar(cmap_mappable, cax=cmap_ax, orientation="horizontal")
cmap_ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
cmap_fig.savefig(os.path.join(out_dir, "energy_colorbar.jpg"))
sebplt.close(cmap_fig)


SITES, PARTICLES = mdfl.production.find_site_and_particle_keys(
    work_dir=work_dir
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
                res[sk][pk] = json_utils.loads(f.read())
        else:
            allsky = mdfl.allsky.AllSky(
                work_dir=os.path.join(work_dir, sk, pk),
                cache_dtype=mdfl.allsky.store.page.dtype(),
            )
            showers = mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
                allsky=allsky,
                azimuth_deg=POINTING["azimuth_deg"],
                zenith_deg=POINTING["zenith_deg"],
                half_angle_deg=POINTING["half_angle_deg"],
                min_num_cherenkov_photons=1e3,
            )
            (
                res[sk][pk]["particle_azimuth_deg"],
                res[sk][pk]["particle_zenith_deg"],
            ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
                cx=showers["particle_cx_rad"],
                cy=showers["particle_cy_rad"],
            )
            res[sk][pk]["particle_energy_GeV"] = showers["particle_energy_GeV"]

            with open(cache_path, "wt") as f:
                f.write(json_utils.dumps(res[sk][pk]))


prng = np.random.Generator(np.random.PCG64(1337))

# hemisphere showing deflections
# ------------------------------
FIELD_OF_VIEW = mdfl.examples.hemisphere_field_of_view()
FIELD_OF_VIEW["wide"]["particles"] = PARTICLES
FIELD_OF_VIEW["narrow"]["particles"] = ["gamma"]


for sk in res:
    sss = atmospheric_cherenkov_response.sites.init(sk)
    mag = mdfl.examples.magnetic_flux(
        earth_magnetic_field_x_muT=sss["earth_magnetic_field_x_muT"],
        earth_magnetic_field_z_muT=sss["earth_magnetic_field_z_muT"],
    )

    for pk in res[sk]:
        showers = res[sk][pk]

        for fk in FIELD_OF_VIEW:
            if pk not in FIELD_OF_VIEW[fk]["particles"]:
                continue

            azimuth_minor_deg = FIELD_OF_VIEW[fk]["azimuth_minor_deg"]
            zenith_minor_deg = FIELD_OF_VIEW[fk]["zenith_minor_deg"]
            rfov = FIELD_OF_VIEW[fk]["rfov"]
            print(sk, pk, fk)

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

            rgbas = cmap_mappable.to_rgba(showers["particle_energy_GeV"])
            rgbas[:, 3] = ALPHA
            _fm = prng.uniform(size=rgbas.shape[0]) <= FRACTION

            sebplt.hemisphere.ax_add_projected_points_with_colors(
                ax=ax,
                azimuths_deg=showers["particle_azimuth_deg"][_fm],
                zeniths_deg=showers["particle_zenith_deg"][_fm],
                half_angle_deg=0.25 * POINTING["half_angle_deg"],
                rgbas=rgbas[_fm],
            )

            sebplt.hemisphere.ax_add_projected_circle(
                ax=ax,
                azimuth_deg=POINTING["azimuth_deg"],
                zenith_deg=POINTING["zenith_deg"],
                half_angle_deg=POINTING["half_angle_deg"],
                linewidth=1.0,
                color="black",
                fill=False,
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
                    "{:s}_{:s}_{:s}.jpg".format(sk, pk, fk),
                )
            )
            sebplt.close(fig)
