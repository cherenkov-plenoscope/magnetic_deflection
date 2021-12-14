#!/usr/bin/python
import sys
import os
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import matplotlib

argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "cherenkov_density_histogram")
os.makedirs(out_dir, exist_ok=True)

CFG = mdfl.read_config(work_dir=work_dir)
PLT = CFG["plotting"]

matplotlib.rcParams["mathtext.fontset"] = PLT["rcParams"]["mathtext.fontset"]
matplotlib.rcParams["font.family"] = PLT["rcParams"]["font.family"]

FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}
AXSPAN = [0.175, 0.2, 0.8, 0.75]

"""
email: 2021-12-06

Werner:
    Es wäre hilfreich (zumindest für Gammas) auch die ganz normalen
    Photonen-Dichte-Plots zu haben; sollte ja einfach sein?

Sebastian:
    Der normale Plot ist die Dichte rho(r) abhaengig vom radius r zum
    Zentrum des Pools auf der x-y-Ebene?
    Wobei bei uns das Zentrum auf der x-y-Ebene durch die Mediane in
    x und y bestimmt ist?
    Ja, das mache ich mal.
"""

# shower_statistics = mdfl.read_statistics(work_dir=work_dir)


def area_in_bins(bin_edges):
    num_bins = len(bin_edges) - 1
    area = np.zeros(num_bins)
    for bb in range(num_bins):
        start = bin_edges[bb]
        stop = bin_edges[bb + 1]
        A = (stop ** 2 - start ** 2) * np.pi
        area[bb] = A
    return area


def recarray_pick_hsitogram(hist_recarray, key_format_str, num_bins):
    num_shower = hist_recarray.shape[0]
    out = np.zeros((num_shower, num_bins))
    for idx in range(num_bins):
        key = key_format_str.format(idx)
        out[:, idx] = hist_recarray[key]
    return out


rs = {}
rs["r"] = {}
rs["r"]["label"] = r"$r$"
rs["r"]["unit"] = r"m"
rs["r"]["bin_edges"] = np.array(CFG["config"]["statistics_r_bin_edges"])
rs["r"]["num_bins"] = len(rs["r"]["bin_edges"]) - 1
rs["r"]["bin_areas"] = area_in_bins(bin_edges=rs["r"]["bin_edges"])
rs["r"]["inverse_area_unit"] = r"m$^{-2}$"

rs["theta"] = {}
rs["theta"]["label"] = r"$\theta$"
rs["theta"]["unit"] = r"(1$^\circ$)"
rs["theta"]["bin_edges"] = np.array(
    CFG["config"]["statistics_theta_bin_edges_deg"]
)
rs["theta"]["num_bins"] = len(rs["theta"]["bin_edges"]) - 1
rs["theta"]["bin_areas"] = area_in_bins(bin_edges=rs["theta"]["bin_edges"])
rs["theta"]["inverse_area_unit"] = r"$(1^\circ)^{-2}$"

for skey in CFG["sites"]:
    for pkey in CFG["particles"]:
        for rkey in rs:
            # spstats = shower_statistics[skey][pkey]

            spstats = mdfl.recarray_io.read_from_tar(
                "__test_mdfl/map/namibia/gamma/000003_statistics.recarray.tar"
            )

            fig = sebplt.figure(FIGSIZE)
            ax = sebplt.add_axes(fig, AXSPAN)
            num_photons = recarray_pick_hsitogram(
                hist_recarray=spstats,
                key_format_str="cherenkov_" + rkey + "_bin_{:06d}",
                num_bins=rs[rkey]["num_bins"],
            )
            p16_num_photons = np.percentile(num_photons, q=16.0, axis=0)
            p50_num_photons = np.percentile(num_photons, q=50.0, axis=0)
            p84_num_photons = np.percentile(num_photons, q=84.0, axis=0)

            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=rs[rkey]["bin_edges"],
                bincounts=p50_num_photons / rs[rkey]["bin_areas"],
                bincounts_upper=p84_num_photons / rs[rkey]["bin_areas"],
                bincounts_lower=p16_num_photons / rs[rkey]["bin_areas"],
                linestyle="-",
                linecolor="k",
                linealpha=1.0,
                face_color="k",
                face_alpha=0.1,
            )
            ax.semilogy()
            ax.set_xlabel(
                rs[rkey]["label"]
                + PLT["label_unit_seperator"]
                + rs[rkey]["unit"]
            )
            ax.set_xlim(
                [min(rs[rkey]["bin_edges"]), max(rs[rkey]["bin_edges"])]
            )
            ax.set_ylabel(
                "density"
                + PLT["label_unit_seperator"]
                + rs[rkey]["inverse_area_unit"]
            )
            filename = "{:s}_{:s}_{:s}".format(skey, pkey, rkey)
            filepath = os.path.join(out_dir, filename)
            fig.savefig(filepath + ".jpg")
            sebplt.close_figure(fig)