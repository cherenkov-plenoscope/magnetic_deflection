import svg_cartesian_plot as svgplt
import spherical_histogram
import numpy as np
import binning_utils
import copy

from .. import allsky


def ax_add_sky(
    ax,
    sky_vertices,
    sky_faces,
    sky_intensity,
    colormap,
    fill_opacity=1.0,
    sky_mask=None,
    sky_mask_color=svgplt.color.css("red"),
):
    mesh_look = svgplt.hemisphere.init_mesh_look(
        num_faces=len(sky_faces),
        stroke=None,
        stroke_opacity=1.0,
        fill=svgplt.color.css("black"),
        fill_opacity=fill_opacity,
    )

    for i in range(len(sky_faces)):
        mesh_look["faces_fill"][i] = colormap(sky_intensity[i])
        mesh_look["faces_stroke"][i] = colormap(sky_intensity[i])
        if sky_mask is not None:
            if sky_mask[i]:
                mesh_look["faces_stroke"][i] = sky_mask_color
                mesh_look["faces_stroke_width"][i] *= 4.0

    svgplt.hemisphere.ax_add_mesh(
        ax=ax,
        vertices=sky_vertices,
        faces=sky_faces,
        max_radius=1.0,
        **mesh_look,
    )


def ax_add_fov(
    ax,
    azimuth_rad,
    zenith_rad,
    half_angle_rad,
    **kwargs,
):
    fov_ring_verts_uxyz = allsky.viewcone.make_ring(
        half_angle_rad=half_angle_rad,
        endpoint=True,
        fn=137,
    )
    fov_ring_verts_uxyz = allsky.viewcone.rotate(
        vertices_uxyz=fov_ring_verts_uxyz,
        azimuth_rad=azimuth_rad,
        zenith_rad=zenith_rad,
        mount="cable_robot",
    )
    svgplt.ax_add_path(
        ax=ax,
        xy=fov_ring_verts_uxyz[:, 0:2],
        **kwargs,
    )


def ax_add_energy_bar(
    ax,
    bin_edges,
    power_slope,
    start,
    stop,
    font_size,
    **kwargs,
):
    yy = np.linspace(0, 1, len(bin_edges))
    for i in range(len(yy)):
        y = yy[i]
        msg = "{: 8.2f}".format(bin_edges[i])
        svgplt.ax_add_line(ax, xy_start=[0, y], xy_stop=[0.5, y], **kwargs)
        kwargs_text = copy.deepcopy(kwargs)
        kwargs_text.pop("stroke", None)
        if np.mod(i, 2) == 0:
            svgplt.ax_add_text(
                ax=ax,
                xy=[1, y - 0.01],
                text=msg,
                font_family="math",
                font_size=font_size,
                **kwargs_text,
            )
    ystart = binning_utils.power.spacing(
        start=bin_edges[0],
        stop=bin_edges[-1],
        x=start,
        power_slope=power_slope,
    )
    ystop = binning_utils.power.spacing(
        start=bin_edges[0],
        stop=bin_edges[-1],
        x=stop,
        power_slope=power_slope,
    )
    svgplt.ax_add_path(
        ax,
        xy=[
            [0, ystart],
            [0.5, ystart],
            [0.5, ystop],
            [0, ystop],
        ],
        fill=svgplt.color.css("red"),
        fill_opacity=1.0,
        **kwargs,
    )

    svgplt.ax_add_line(ax, xy_start=[0, 0], xy_stop=[0, 1], **kwargs)
