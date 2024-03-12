import svg_cartesian_plot as svgplt
import spherical_histogram
import numpy as np
import binning_utils
import copy

from .. import allsky


def ax_add_sky(
    ax,
    sky_valuesertices,
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
        vertices=sky_valuesertices,
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
    query_color,
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
        fill=svgplt.color.css(query_color),
        fill_opacity=1.0,
        **kwargs,
    )

    svgplt.ax_add_line(ax, xy_start=[0, 0], xy_stop=[0, 1], **kwargs)


def plot_query(
    path,
    query,
    skymap,
    sky_values,
    sky_values_min,
    sky_values_max,
    sky_values_label,
    sky_values_scale="log",
    sky_mask=None,
    sky_mask_color="orange",
    colormap_name="viridis",
    query_color="white",
    num_pixel=1280,
):
    # estimate p50 of p50 brightest faces
    _sky_p50_mask = binning_utils.mask_fewest_bins_to_contain_quantile(
        bin_counts=sky_values,
        quantile=0.5,
    )
    _sky_brightest_faces = sky_values[_sky_p50_mask]
    if len(_sky_brightest_faces) > 0:
        sky_values_p50 = np.quantile(_sky_brightest_faces, q=0.5)
    else:
        sky_values_p50 = 0.0

    if sky_values_scale == "log":
        intensity_scale = svgplt.scaling.log(base=10)
    elif sky_values_scale == "unity":
        intensity_scale = svgplt.scaling.unity()
    else:
        raise ValueError(
            "Unknown sky_values_scale '{:s}'.".format(sky_values_scale)
        )

    colormap = svgplt.color.Map(
        name=colormap_name,
        start=sky_values_min,
        stop=sky_values_max,
        func=intensity_scale,
    )

    fig = svgplt.Fig(cols=(3 * num_pixel) // 2, rows=num_pixel)

    font_size = 15 * num_pixel / 1280
    stroke_width = num_pixel / 1280

    ax = svgplt.hemisphere.Ax(fig=fig)
    ax["span"] = (0.0, 0.0, 1 / (3 / 2), 1)

    axw = svgplt.Ax(fig=fig)
    axw["span"] = (0.7, 0.1, 0.025, 0.8)
    axw["yscale"] = colormap.func

    axe = svgplt.Ax(fig=fig)
    axe["span"] = (0.85, 0.1, 0.05, 0.8)

    ax_add_sky(
        ax=ax,
        sky_valuesertices=skymap.binning["sky"].vertices,
        sky_faces=skymap.binning["sky"].faces,
        sky_intensity=sky_values,
        colormap=colormap,
        fill_opacity=1.0,
        sky_mask=sky_mask,
        sky_mask_color=svgplt.color.css(sky_mask_color),
    )

    ax_add_fov(
        ax=ax,
        azimuth_rad=query["azimuth_rad"],
        zenith_rad=query["zenith_rad"],
        half_angle_rad=query["half_angle_rad"],
        stroke=svgplt.color.css(query_color),
        stroke_width=4 * stroke_width,
        fill=None,
    )

    ax_add_energy_bar(
        ax=axe,
        bin_edges=skymap.binning["energy"]["edges"],
        power_slope=skymap.config["energy_power_slope"],
        start=query["energy_start_GeV"],
        stop=query["energy_stop_GeV"],
        font_size=3 * font_size,
        stroke_width=1.5 * stroke_width,
        query_color=query_color,
        stroke=svgplt.color.css("black"),
    )

    svgplt.color.ax_add_colormap(
        ax=axw,
        colormap=colormap,
        fn=128,
        orientation="vertical",
    )
    svgplt.color.ax_add_colormap_ticks(
        ax=axw,
        colormap=colormap,
        num=6,
        orientation="vertical",
        fill=svgplt.color.css("black"),
        stroke=None,
        stroke_width=1.5 * stroke_width,
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.ax_add_line(
        ax=axw,
        xy_start=[-0.5, sky_values_p50],
        xy_stop=[0, sky_values_p50],
        stroke=svgplt.color.css("black"),
        stroke_width=5 * stroke_width,
    )

    svgplt.hemisphere.ax_add_grid(
        ax=ax,
        stroke=svgplt.color.css("white"),
        stroke_opacity=1.0,
        stroke_width=0.3 * stroke_width,
        font_size=3.0 * font_size,
    )
    svgplt.ax_add_text(
        ax=axe,
        xy=[-5, -0.075],
        text=sky_values_label,
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.ax_add_text(
        ax=axe,
        xy=[0.0, -0.075],
        text="energy / GeV",
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.ax_add_text(
        ax=ax,
        xy=[-1.0, -0.85],
        text="{:s}".format(skymap.config["site"]["key"]),
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.ax_add_text(
        ax=ax,
        xy=[-1.0, -0.95],
        text="{:s}".format(skymap.config["particle"]["key"]),
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=3 * font_size,
    )

    svgplt.fig_write(fig=fig, path=path)


def plot_exposure(path, skymap, enebin, num_pixel=1280):
    sky_num_primaries = skymap.map_exposure()[enebin]
    sky_num_primaries_per_sr = (
        sky_num_primaries / skymap.binning["sky"].faces_solid_angles
    )

    colormap = svgplt.color.Map(
        name="viridis",
        start=0.0,
        stop=np.max(sky_num_primaries_per_sr),
        func=svgplt.scaling.unity(),
    )

    fig = svgplt.Fig(cols=(3 * num_pixel) // 2, rows=num_pixel)

    font_size = 15 * num_pixel / 1280
    stroke_width = num_pixel / 1280

    ax = svgplt.hemisphere.Ax(fig=fig)
    ax["span"] = (0.0, 0.0, 1 / (3 / 2), 1)

    axw = svgplt.Ax(fig=fig)
    axw["span"] = (0.7, 0.1, 0.025, 0.8)
    axw["yscale"] = colormap.func

    ax_add_sky(
        ax=ax,
        sky_valuesertices=skymap.binning["sky"].vertices,
        sky_faces=skymap.binning["sky"].faces,
        sky_intensity=sky_num_primaries_per_sr,
        colormap=colormap,
        fill_opacity=1.0,
        sky_mask=None,
        sky_mask_color=None,
    )
    svgplt.hemisphere.ax_add_grid(
        ax=ax,
        stroke=svgplt.color.css("white"),
        stroke_opacity=1.0,
        stroke_width=0.3 * stroke_width,
        font_size=3.0 * font_size,
    )
    svgplt.color.ax_add_colormap(
        ax=axw,
        colormap=colormap,
        fn=128,
        orientation="vertical",
    )
    svgplt.color.ax_add_colormap_ticks(
        ax=axw,
        colormap=colormap,
        num=6,
        orientation="vertical",
        fill=svgplt.color.css("black"),
        stroke=None,
        stroke_width=1.5 * stroke_width,
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.ax_add_text(
        ax=ax,
        xy=[0.9, -0.9],
        text="primary density / (sr)\u207b\u00b9",
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=3 * font_size,
    )
    svgplt.fig_write(fig=fig, path=path)


def plot_map(path, skymap, enebin, num_pixel=1280):
    p2c = skymap.map_primary_to_cherenkov_normalized_per_sr()[enebin]

    colormap = svgplt.color.Map(
        name="viridis",
        start=1e3,
        stop=1e8,
        func=svgplt.scaling.log(base=10),
    )

    fig = svgplt.Fig(cols=(3 * num_pixel) // 2, rows=num_pixel)

    font_size = 15 * num_pixel / 1280
    stroke_width = num_pixel / 1280

    ax = svgplt.Ax(fig=fig)
    ax["span"] = (0.1, 0.1, 0.8 / (3 / 2), 0.8)

    svgplt.ax_add_pcolormesh(
        ax=ax,
        z=p2c,
        colormap=colormap,
        fill_opacity=1.0,
    )
    svgplt.fig_write(fig=fig, path=path)
