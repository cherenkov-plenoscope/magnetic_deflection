import argparse
import os
import binning_utils
import sebastians_matplotlib_addons as sebplt
import atmospheric_cherenkov_response
from atmospheric_cherenkov_response import plot
import numpy as np
import spherical_histogram
import spherical_coordinates
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

PLT = atmospheric_cherenkov_response.plot.config()
sebplt.matplotlib.rcParams.update(PLT["rcParams"])

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

FIGSIZE = {"rows": 640, "cols": 1280, "fontsize": 1.25}
fig = sebplt.figure(FIGSIZE)
ax = sebplt.add_axes(fig=fig, span=(0.15, 0.25, 0.8, 0.5))
axSqDeg = sebplt.add_axes(
    fig=fig,
    span=(0.15, 0.75, 0.8, 0.0),
    style={"spines": ["bottom"], "axes": ["x"], "grid": False},
)
axSqDeg.xaxis.set_label_position("top")
axSqDeg.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
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
ax.semilogy()
ax.set_xlabel("solid angle / msr")
ax.set_ylabel("intensity / 1")
ax.set_xlim(1e3 * sa_bin["limits"])
fig.savefig(os.path.join(out_dir, "skymap_faces_solid_angles.jpg"))
sebplt.close(fig)


msr = skymap_faces_solid_angles * 1e3
sdq = sau.sr2squaredeg(skymap_faces_solid_angles)
pss = f"faces solid angles: {np.mean(msr):.1f} +- {np.std(msr):.1f} msr"
pss += f", {np.mean(sdq):.1f} +- {np.std(sdq):.1f} (1deg)^2"
pss += f", portal fov {3.25 ** 2 * np.pi:.1f} (1deg)^2"


print(pss)

# 3D
import mpl_toolkits.mplot3d

azim_deg = -63
elev_deg = 11

FIGSIZE = {"rows": 1280, "cols": 1280, "fontsize": 1.25}
fig = sebplt.figure(FIGSIZE)
ax3d = mpl_toolkits.mplot3d.Axes3D(
    fig,
    auto_add_to_figure=False,
    azim=azim_deg,
    elev=elev_deg,
    focal_length=1.0,
    computed_zorder=True,
)
fig.add_axes(ax3d, span=[0.3, 0.3, 0.5, 0.5])


view_cxcycz = spherical_coordinates.az_zd_to_cx_cy_cz(
    azimuth_rad=np.deg2rad(azim_deg),
    zenith_rad=np.deg2rad(90 - elev_deg),
)


def ax3d_add_horizontal(
    ax3d, zd, az_start=0, az_stop=2 * np.pi, radius=1.01, **kwargs
):
    x = []
    y = []
    z = []
    for phi in np.linspace(0, 2 * np.pi, 180, endpoint=True):
        _x, _y, _z = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=phi,
            zenith_rad=zd,
        )

        theta = spherical_coordinates.angle_between_cx_cy_cz(
            _x, _y, _z, view_cxcycz[0], view_cxcycz[1], view_cxcycz[2]
        )

        x.append(_x * radius)
        y.append(_y * radius)
        z.append(_z * radius)

    ax3d.plot(x, y, z, **kwargs)


def ax3d_add_vertical_arc(
    ax3d,
    az,
    zd_start=np.deg2rad(10),
    zd_stop=np.deg2rad(90),
    radius=1.01,
    **kwargs,
):
    x = []
    y = []
    z = []
    for zz in np.linspace(zd_start, zd_stop, 180, endpoint=True):
        _x, _y, _z = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=az,
            zenith_rad=zz,
        )
        x.append(_x * radius)
        y.append(_y * radius)
        z.append(_z * radius)
    ax3d.plot(x, y, z, **kwargs)


light_vector = spherical_coordinates.az_zd_to_cx_cy_cz(
    azimuth_rad=np.deg2rad(-40),
    zenith_rad=np.deg2rad(80),
)
verts = []
vert_colors = []
rgb_color = np.array((1, 1, 1))
for face in skymap_obj["mtl"]["sky"]:
    vvv = [skymap_obj["v"][i] for i in face["v"]]
    nnn = triangle_mesh_io.convert._mesh.normal.make_normal_from_face(
        a=vvv[0],
        b=vvv[1],
        c=vvv[2],
    )
    theta = spherical_coordinates.angle_between_cx_cy_cz(
        nnn[0],
        nnn[1],
        nnn[2],
        light_vector[0],
        light_vector[1],
        light_vector[2],
    )
    verts.append(vvv)

    shade = 0.2 + 0.75 * np.abs(np.cos(theta))
    vert_colors.append(rgb_color * shade)

lineargs = {
    "color": "black",
    "alpha": 0.3,
    "linewidth": 0.25,
}

for zd_deg in np.linspace(10, 90, 9):
    ax3d_add_horizontal(
        ax3d=ax3d,
        zd=np.deg2rad(zd_deg),
        **lineargs,
    )
for az_deg in np.linspace(0, 360, 36, endpoint=False):
    ax3d_add_vertical_arc(
        ax3d=ax3d,
        az=np.deg2rad(az_deg),
        **lineargs,
    )

ax3d.add_collection3d(
    mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        verts,
        color=vert_colors,
    )
)

ax3d.plot(
    xs=[0, 0],
    ys=[0, 0],
    zs=[0, 1.1],
    **lineargs,
)
ax3d.plot(
    xs=[-1, 1],
    ys=[0, 0],
    zs=[0, 0],
    **lineargs,
)
ax3d.plot(
    xs=[0, 0],
    ys=[-1, 1],
    zs=[0, 0],
    **lineargs,
)
ax3d.text(0, 0, 1.1, "Zenith")

ax3d.text(1.1, 0, 0, "North")
ax3d.text(0, -1.2, 0, "East")
ax3d.text(-1.3, 0, 0, "South")

ax3d.set_box_aspect(np.array([1, 1, 0.5]) * 0.5)
ax3d.axes.set_xlim3d([-1, 1])
ax3d.axes.set_ylim3d([-1, 1])
ax3d.axes.set_zlim3d([0, 1])

ax3d.grid(visible=False)
ax3d.set_axis_off()


fig.savefig(os.path.join(out_dir, "skymap_3d.jpg"))
sebplt.close(fig)
