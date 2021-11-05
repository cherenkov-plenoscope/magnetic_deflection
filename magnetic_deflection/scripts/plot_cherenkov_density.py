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

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
figsize = {"rows": 720, "cols": 1280, "fontsize": 1.25}

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot_cherenkov_density")
os.makedirs(out_dir, exist_ok=True)

shower_statistics = mdfl.read_shower_statistics(work_dir=work_dir)
c = mdfl.read_config(work_dir=work_dir)

density_map = {
    "num_photons": {"label": "intensity$\,/\,$1", "lim": [1e2, 1e6],},
    "spread_area_m2": {"label": "area$\,/\,$m$^{2}$", "lim": [1e4, 1e7],},
    "spread_solid_angle_deg2": {
        "label": "solid angle$\,/\,(1^\circ)^{2}$",
        "lim": [1e-1, 1e2],
    },
    "light_field_outer_density": {
        "label": "density$\,/\,$m$^{-2}\,(1^\circ)^{-2}$",
        "lim": [1e-3, 1e2],
    },
}

parmap = {"gamma": "k", "electron": "b", "proton": "r", "helium": "orange"}

# compute density
cherenkov_density = {}
for skey in shower_statistics:
    cherenkov_density[skey] = {}
    for pkey in shower_statistics[skey]:
        sst = shower_statistics[skey][pkey]

        den = {}
        den["num_photons"] = sst["num_photons"]
        den["spread_area_m2"] = (
            np.pi * sst["position_std_major_m"] * sst["position_std_minor_m"]
        )
        den["spread_solid_angle_deg2"] = (
            np.pi
            * np.rad2deg(sst["direction_std_major_rad"])
            * np.rad2deg(sst["direction_std_minor_rad"])
        )
        den["light_field_outer_density"] = sst["num_photons"] / (
            den["spread_solid_angle_deg2"] * den["spread_area_m2"]
        )

        cherenkov_density[skey][pkey] = pd.DataFrame(den).to_records(
            index=False
        )


def make_off_axis_angle_deg(
    cherenkov_direction_med_cx_rad,
    cherenkov_direction_med_cy_rad,
    pointing_azimuth_deg,
    pointing_zenith_deg,
):
    (
        cer_azimuth_deg,
        cer_zenith_deg,
    ) = mdfl.spherical_coordinates._cx_cy_to_az_zd_deg(
        cx=cherenkov_direction_med_cx_rad, cy=cherenkov_direction_med_cy_rad,
    )

    return mdfl.spherical_coordinates._angle_between_az_zd_deg(
        az1_deg=cer_azimuth_deg,
        zd1_deg=cer_zenith_deg,
        az2_deg=pointing_azimuth_deg,
        zd2_deg=pointing_zenith_deg,
    )


num_energy_bins = 60
energy_bin_edges = np.geomspace(1e-1, 1e2, num_energy_bins + 1)


for skey in shower_statistics:
    for dkey in density_map:
        alpha = 0.2

        fig = sebplt.figure(figsize)
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

        for pkey in parmap:
            sst = shower_statistics[skey][pkey]
            cde = cherenkov_density[skey][pkey][dkey]

            off_axis_deg = make_off_axis_angle_deg(
                cherenkov_direction_med_cx_rad=sst["direction_med_cx_rad"],
                cherenkov_direction_med_cy_rad=sst["direction_med_cy_rad"],
                pointing_azimuth_deg=c["pointing"]["azimuth_deg"],
                pointing_zenith_deg=c["pointing"]["zenith_deg"],
            )
            on_axis_mask = (
                off_axis_deg
                <= 2
                * c["particles"][pkey]["magnetic_deflection_max_off_axis_deg"]
            )

            for e_bin in range(num_energy_bins):
                E_start = energy_bin_edges[e_bin]
                E_stop = energy_bin_edges[e_bin + 1]
                E_mask = np.logical_and(
                    sst["particle_energy_GeV"] >= E_start,
                    sst["particle_energy_GeV"] < E_stop,
                )
                mask = np.logical_and(E_mask, on_axis_mask)
                num_samples = np.sum(mask)

                print(skey, pkey, e_bin, num_samples)

                if num_samples:
                    dens_avg = np.percentile(cde[mask], 50)
                    dens_std_plus = np.percentile(cde[mask], 50 + 68 / 2)
                    dens_std_minus = np.percentile(cde[mask], 50 - 68 / 2)

                    d_start = dens_std_minus
                    d_stop = dens_std_plus
                    ax.fill(
                        [E_start, E_start, E_stop, E_stop],
                        [d_start, d_stop, d_stop, d_start],
                        color=parmap[pkey],
                        alpha=alpha,
                        linewidth=0.0,
                    )
                    ax.plot(
                        [E_start, E_stop],
                        [dens_avg, dens_avg],
                        color=parmap[pkey],
                        alpha=1.0,
                    )
        ax.loglog()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("energy$\,/\,$GeV")
        ax.set_ylabel(density_map[dkey]["label"])
        ax.set_xlim([min(energy_bin_edges), max(energy_bin_edges)])
        ax.set_ylim(density_map[dkey]["lim"])
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(skey, dkey)))
        plt.close(fig)
