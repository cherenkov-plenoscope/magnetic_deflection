import svg_cartesian_plot as svgplt
import spherical_histogram
import spherical_coordinates
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


def ax_add_background_sky(
    ax,
    r=1.0,
    fn=180,
    **kwargs,
):
    xy = []
    for phi in np.linspace(0.0, 2 * np.pi, fn, endpoint=False):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xy.append([x, y])
    xy = np.asarray(xy)

    svgplt.ax_add_path(
        ax=ax,
        xy=xy,
        **kwargs,
    )


def ax_add_cross(
    ax,
    azimuth_rad,
    zenith_rad,
    min_half_angle_rad,
    max_half_angle_rad,
    **kwargs,
):
    assert min_half_angle_rad < max_half_angle_rad
    sph2xyz = spherical_coordinates.az_zd_to_cx_cy_cz
    min_ha = min_half_angle_rad
    max_ha = max_half_angle_rad

    for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        xyz_start = sph2xyz(azimuth_rad=phi, zenith_rad=min_ha)
        xyz_stop = sph2xyz(azimuth_rad=phi, zenith_rad=max_ha)

        verts = np.c_[np.asarray(xyz_start), np.asarray(xyz_stop)].T

        verts = allsky.viewcone.rotate(
            vertices_uxyz=verts,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            mount="altitude_azimuth",
        )
        svgplt.ax_add_path(
            ax=ax,
            xy=verts[:, 0:2],
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


def estimate_median_from_only_the_50percent_largest_bins(bin_counts):
    p50_mask = binning_utils.mask_fewest_bins_to_contain_quantile(
        bin_counts=bin_counts,
        quantile=0.5,
    )
    fullest_bins = bin_counts[p50_mask]
    if len(fullest_bins) > 0:
        return np.quantile(fullest_bins, q=0.5)
    else:
        return 0.0


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
    sky_values_p50 = estimate_median_from_only_the_50percent_largest_bins(
        bin_counts=sky_values
    )

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
        sky_vertices=skymap.binning["sky"].vertices,
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
        sky_vertices=skymap.binning["sky"].vertices,
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
        text="primary density / sr" + svgplt.text.superscript("-1"),
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


def plot_draw(path, skymap, result, debug):
    sky_cherenkov_p50_per_sr = (
        estimate_median_from_only_the_50percent_largest_bins(
            bin_counts=debug["sky_cherenkov_per_sr"]
        )
    )

    query_color = "white"
    num_pixel = 1280

    vmin_cherenkov_density_per_sr = 1e3
    vmax_cherenkov_density_per_sr = 1e8

    colormap = svgplt.color.Map(
        name="viridis",
        start=vmin_cherenkov_density_per_sr,
        stop=vmax_cherenkov_density_per_sr,
        func=svgplt.scaling.log(base=10),
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

    ax_add_background_sky(
        ax=ax,
        fill=svgplt.color.css("black"),
        fill_opacity=1.0,
        stroke=None,
    )

    mesh_look = svgplt.hemisphere.init_mesh_look(
        num_faces=len(skymap.binning["sky"].faces),
        stroke=None,
        stroke_opacity=1.0,
        fill=svgplt.color.css("black"),
        fill_opacity=1.0,
    )
    for i in range(len(skymap.binning["sky"].faces)):
        _face_value = debug["sky_cherenkov_per_sr"][i]
        mesh_look["faces_fill"][i] = colormap(_face_value)
        if debug["sky_above_threshold_mask"][i]:
            mesh_look["faces_stroke"][i] = colormap(_face_value)
        else:
            mesh_look["faces_stroke"][i] = svgplt.color.css("black")

        if debug["sky_draw_mask"][i]:
            mesh_look["faces_stroke"][i] = svgplt.color.css("orange")
            if i == debug["face"]:
                mesh_look["faces_stroke_opacity"][i] = 1.0
                mesh_look["faces_stroke_width"][i] *= 4.0
            else:
                mesh_look["faces_stroke_opacity"][i] = 0.67
                mesh_look["faces_stroke_width"][i] *= 2.0

    svgplt.hemisphere.ax_add_mesh(
        ax=ax,
        vertices=skymap.binning["sky"].vertices,
        faces=skymap.binning["sky"].faces,
        max_radius=1.0,
        **mesh_look,
    )

    ax_add_fov(
        ax=ax,
        azimuth_rad=debug["parameters"]["azimuth_rad"],
        zenith_rad=debug["parameters"]["zenith_rad"],
        half_angle_rad=debug["parameters"]["half_angle_rad"],
        stroke=svgplt.color.css(query_color),
        stroke_width=4 * stroke_width,
        fill=None,
    )

    ax_add_energy_bar(
        ax=axe,
        bin_edges=skymap.binning["energy"]["edges"],
        power_slope=skymap.config["energy_power_slope"],
        start=debug["parameters"]["energy_start_GeV"],
        stop=debug["parameters"]["energy_stop_GeV"],
        font_size=3 * font_size,
        stroke_width=1.5 * stroke_width,
        query_color=query_color,
        stroke=svgplt.color.css("black"),
    )

    ax_add_cross(
        ax=ax,
        azimuth_rad=result["particle_azimuth_rad"],
        zenith_rad=result["particle_zenith_rad"],
        min_half_angle_rad=np.deg2rad(1.0),
        max_half_angle_rad=np.deg2rad(2.0),
        stroke=svgplt.color.css("red"),
        stroke_width=4 * stroke_width,
        fill=None,
    )
    ax_add_fov(
        ax=ax,
        azimuth_rad=result["particle_azimuth_rad"],
        zenith_rad=result["particle_zenith_rad"],
        half_angle_rad=np.deg2rad(1.5),
        stroke=svgplt.color.css("red"),
        stroke_width=4 * stroke_width,
        fill=None,
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
        xy_start=[-0.5, sky_cherenkov_p50_per_sr],
        xy_stop=[0, sky_cherenkov_p50_per_sr],
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
        text="Cherenkov density / sr" + svgplt.text.superscript("-1"),
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
