import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import numpy as np
import spherical_histogram
import solid_angle_utils as sau
import triangle_mesh_io


parser = argparse.ArgumentParser(
    prog="plot_solid_angle_distribution_of_skymap_bins.py",
    description=("Make histogram plot solid angle distribution."),
)
parser.add_argument(
    "--skymap_obj",
    metavar="PATH",
    type=str,
    help="Path of the skymap wavefront",
)
parser.add_argument(
    "--out_dir",
    metavar="STRING",
    type=str,
    help="Directory to write figures to.",
)

args = parser.parse_args()

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

with open(args.skymap_obj, "rt") as f:
    skymap_obj = triangle_mesh_io.obj.loads(f.read())
    (
        skymap_vertices,
        skymap_faces,
    ) = spherical_histogram.mesh.obj_to_vertices_and_faces(skymap_obj)

skymap_faces_solid_angles = spherical_histogram.mesh.estimate_solid_angles(
    vertices=skymap_vertices,
    faces=skymap_faces,
)

num_bins = int(np.ceil(np.sqrt(len(skymap_faces))))
sa_bin = binning_utils.Binning(
    bin_edges=np.linspace(0.0, sau.squaredeg2sr(25), num_bins)
)

bin_counts = np.histogram(skymap_faces_solid_angles, bins=sa_bin["edges"])[0]

FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}
fig = sebplt.figure(FIGSIZE)
ax = sebplt.add_axes(fig=fig, span=(0.175, 0.45, 0.8, 0.5))
axSqDeg = sebplt.add_axes(
    fig=fig,
    span=(0.175, 0.2, 0.8, 0.05),
    style={"spines": ["bottom"], "axes": ["x"], "grid": False},
)
axSqDeg.set_xlim(sau.sr2squaredeg(sa_bin["limits"]))
axSqDeg.set_xlabel("solid angle / ($1^{\circ{}})^2$")

sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=sa_bin["edges"] * 1e3,
    bincounts=bin_counts,
    linestyle="-",
    linecolor="black",
    linealpha=1.0,
    face_color="black",
    face_alpha=0.25,
    label=None,
    draw_bin_walls=True,
)
ax.set_xlabel("solid angle / msr")
ax.set_ylabel("intensity / 1")
ax.set_xlim(1e3 * sa_bin["limits"])
fig.savefig(os.path.join(out_dir, "skymap_faces_solid_angles.jpg"))
sebplt.close(fig)
