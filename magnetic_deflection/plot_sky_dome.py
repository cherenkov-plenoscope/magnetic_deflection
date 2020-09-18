import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as plt_patches
from matplotlib import colors as plt_colors


def add_circle(ax, x, y, r, linewidth, color, alpha):
    phis = np.linspace(0, 2*np.pi, 1001)
    xs = r*np.cos(phis)
    ys = r*np.sin(phis)
    ax.plot(xs, ys, linewidth=linewidth, color=color, alpha=alpha)


def add_points_in_half_dome(
    ax,
    azimuths_deg,
    zeniths_deg,
    point_diameter,
    color=None,
    alpha=None,
    rgbas=None,
):
    zeniths = np.deg2rad(zeniths_deg)
    azimuths = np.deg2rad(azimuths_deg)

    proj_radii = np.sin(zeniths)
    proj_x = np.cos(azimuths)*proj_radii
    proj_y = np.sin(azimuths)*proj_radii

    if rgbas is not None:
        _colors = rgbas[:, 0:3]
        _alphas = rgbas[:, 3]
    else:
        assert color is not None
        assert alpha is not None
        _colors = [color for i in range(len(zeniths))]
        _alphas = [alpha for i in range(len(zeniths))]

    for i in range(len(zeniths)):
        e1 = plt_patches.Ellipse(
            (proj_x[i], proj_y[i]),
            width=point_diameter*np.cos(zeniths[i]),
            height=point_diameter,
            angle=np.rad2deg(azimuths[i]),
            linewidth=0,
            fill=True,
            zorder=2,
            facecolor=_colors[i],
            alpha=_alphas[i])
        ax.add_patch(e1)


def add_grid_in_half_dome(
    ax,
    azimuths_deg,
    zeniths_deg,
    linewidth,
    color,
    alpha,
    draw_lower_horizontal_edge_deg=None,
):
    zeniths = np.deg2rad(zeniths_deg)
    proj_radii = np.sin(zeniths)
    for i in range(len(zeniths)):
        add_circle(
            ax=ax,
            x=0,
            y=0,
            r=proj_radii[i],
            linewidth=linewidth*np.cos(zeniths[i]),
            color=color,
            alpha=alpha)

    azimuths = np.deg2rad(azimuths_deg)
    for a in range(len(azimuths)):
        for z in range(len(zeniths)):
            if z == 0:
                continue
            r_start = np.sin(zeniths[z-1])
            r_stop = np.sin(zeniths[z])
            start_x = r_start*np.cos(azimuths[a])
            start_y = r_start*np.sin(azimuths[a])
            stop_x = r_stop*np.cos(azimuths[a])
            stop_y = r_stop*np.sin(azimuths[a])
            ax.plot(
                [start_x, stop_x],
                [start_y, stop_y],
                color=color,
                linewidth=linewidth*np.cos(zeniths[z-1]),
                alpha=alpha)

    if draw_lower_horizontal_edge_deg is not None:
        zd_edge = np.deg2rad(draw_lower_horizontal_edge_deg)
        r = np.sin(zd_edge)
        ax.plot(
            [-2/3*r, 0],
            [-r, -r],
            color=color,
            linewidth=linewidth,
            alpha=alpha)
        add_circle(
            ax=ax,
            x=0,
            y=0,
            r=r,
            linewidth=linewidth,
            color=color,
            alpha=alpha)


def add_ticklabels_in_half_dome(
    ax,
    azimuths_deg,
    rfov=1.0,
    fmt=r"{:1.0f}$^\circ$",
):
    xshift = -0.1*rfov
    yshift = -0.05*rfov

    azimuths = np.deg2rad(azimuths_deg)
    azimuth_deg_strs = [fmt.format(az) for az in azimuths_deg]
    xs = rfov*np.cos(azimuths) + xshift
    ys = rfov*np.sin(azimuths) + yshift
    for a in range(len(azimuths)):
        ax.text(
            x=xs[a],
            y=ys[a],
            s=azimuth_deg_strs[a])
