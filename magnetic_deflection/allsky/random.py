import numpy as np
import corsika_primary
import svg_cartesian_plot as svgplt
import solid_angle_utils

from . import hemisphere
from . import viewcone
from .. import spherical_coordinates
from ..version import __version__


def guess_energy_factor(start, stop, e, start_factor=1e-2, stop_factor=1e-1):
    # power spectrum is: -2.0
    lstart = np.log10(start)
    lstop = np.log10(stop)
    le = np.log10(e)
    return np.interp(x=le, xp=[lstart, lstop], fp=[start_factor, stop_factor])


def guess_energy_factor_from_allsky_deflection(allsky_deflection, energy_GeV):
    return guess_energy_factor(
        start=allsky_deflection.config["binning"]["energy"]["start_GeV"],
        stop=allsky_deflection.config["binning"]["energy"]["stop_GeV"],
        e=energy_GeV,
    )


def estimate_cluster_labels_for_matches(matches, eps_deg, min_samples):
    _cz = spherical_coordinates.restore_cz(
        cx=matches["particle_cx_rad"], cy=matches["particle_cy_rad"]
    )
    particle_cxcycz = np.c_[
        matches["particle_cx_rad"], matches["particle_cy_rad"], _cz
    ]

    eps_rad = np.deg2rad(eps_deg)
    labels = hemisphere.cluster(
        vertices=particle_cxcycz,
        eps=eps_rad,
        min_samples=min_samples,
    )
    return labels


def apply_cluster_labels(matches, cluster_labels, weights=None):
    mask = cluster_labels >= 0  # everything but outliers
    if weights is None:
        return matches[mask]
    else:
        return matches[mask], weights[mask]


def draw_particle_direction_with_cone(
    prng,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    energy_GeV,
    shower_spread_half_angle_deg,
    min_num_cherenkov_photons,
    allsky_deflection,
    energy_factor=None,
    max_iterations=1000 * 1000,
):
    if energy_factor is None:
        energy_factor = guess_energy_factor_from_allsky_deflection(
            allsky_deflection=allsky_deflection, energy_GeV=energy_GeV
        )

    debug = {"method_key": "cone"}
    debug["version"] = __version__
    debug["parameters"] = {
        "azimuth_deg": azimuth_deg,
        "zenith_deg": zenith_deg,
        "half_angle_deg": half_angle_deg,
        "energy_GeV": energy_GeV,
        "energy_factor": energy_factor,
        "shower_spread_half_angle_deg": shower_spread_half_angle_deg,
        "min_num_cherenkov_photons": min_num_cherenkov_photons,
    }

    max_zenith_distance_deg = allsky_deflection.config["binning"]["direction"][
        "particle_max_zenith_distance_deg"
    ]
    debug["max_zenith_distance_deg"] = max_zenith_distance_deg

    try:
        (
            matches,
            direction_weights,
            energy_weights,
        ) = allsky_deflection.query_cherenkov_ball(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=half_angle_deg,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
            weights=True,
        )
    except RuntimeError as err:
        assert "Not enough population" in err.__str__()
        result = {"cutoff": True}
        return result, debug

    if np.sum(energy_weights) < 0.1:
        result = {"cutoff": True}
        return result, debug

    # cluster
    # -------
    min_samples = max([10, int(np.sqrt(len(matches)))])
    debug["cluster_min_samples"] = min_samples
    cluster_labels = estimate_cluster_labels_for_matches(
        matches=matches,
        eps_deg=(1 / 3) * shower_spread_half_angle_deg,
        min_samples=min_samples,
    )
    debug["cluster_labels"] = cluster_labels
    matches, energy_weights = apply_cluster_labels(
        matches=matches, weights=energy_weights, cluster_labels=cluster_labels
    )
    if len(matches) == 0:
        result = {"cutoff": True}
        return result, debug

    # cone
    # ----
    avg_particle_cx_rad, std_particle_cx_rad = weighted_avg_and_std(
        values=matches["particle_cx_rad"], weights=energy_weights
    )
    avg_particle_cy_rad, std_particle_cy_rad = weighted_avg_and_std(
        values=matches["particle_cy_rad"], weights=energy_weights
    )

    (
        avg_particle_azimuth_deg,
        avg_particle_zenith_deg,
    ) = spherical_coordinates._cx_cy_to_az_zd_deg(
        cx=avg_particle_cx_rad, cy=avg_particle_cy_rad
    )

    debug["query_ball"] = {
        "particle_cx_rad": matches["particle_cx_rad"],
        "particle_cy_rad": matches["particle_cy_rad"],
        "energy_weights": energy_weights,
    }
    debug["average"] = {
        "particle_azimuth_deg": avg_particle_azimuth_deg,
        "particle_zenith_deg": avg_particle_zenith_deg,
    }

    assert avg_particle_zenith_deg <= max_zenith_distance_deg

    half_angle_thrown_rad = np.deg2rad(
        shower_spread_half_angle_deg
    ) + np.deg2rad(half_angle_deg)
    half_angle_thrown_deg = np.rad2deg(half_angle_thrown_rad)

    debug["cone"] = {}
    debug["cone"]["half_angle_thrown_deg"] = half_angle_thrown_deg

    if (
        avg_particle_zenith_deg + half_angle_thrown_deg
        <= max_zenith_distance_deg
    ):
        debug["cone"]["is_truncated_by_max_zenith_distance"] = False
        # A full cone:
        solid_angle_thrown_sr = solid_angle_utils.cone.solid_angle(
            half_angle_rad=half_angle_thrown_rad
        )
        debug["cone"]["solid_angle_sr"] = solid_angle_thrown_sr
    else:
        debug["cone"]["is_truncated_by_max_zenith_distance"] = True
        # A truncated cone:
        cone_vertices = viewcone.make_ring(
            half_angle_deg=np.rad2deg(half_angle_thrown_rad),
            endpoint=True,
            fn=137,
        )
        cone_vertices = viewcone.rotate(
            vertices_uxyz=cone_vertices,
            azimuth_deg=avg_particle_azimuth_deg,
            zenith_deg=avg_particle_zenith_deg,
        )
        cone_vertices = viewcone.limit_zenith_distance(
            vertices_uxyz=cone_vertices,
            max_zenith_distance_deg=max_zenith_distance_deg,
        )
        cone_faces = hemisphere.make_faces(vertices=cone_vertices)
        solid_angle_thrown_sr = hemisphere.estimate_solid_angles(
            vertices=cone_vertices, faces=cone_faces
        )
        debug["cone"]["vertices"] = cone_vertices
        debug["cone"]["faces"] = cone_faces
        debug["cone"]["solid_angle_sr"] = solid_angle_thrown_sr

    (
        particle_azimuth_rad,
        particle_zenith_rad,
    ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
        prng=prng,
        azimuth_rad=np.deg2rad(avg_particle_azimuth_deg),
        zenith_rad=np.deg2rad(avg_particle_zenith_deg),
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=half_angle_thrown_rad,
        max_zenith_rad=np.deg2rad(corsika_primary.MAX_ZENITH_DEG),
        max_iterations=max_iterations,
    )

    result = {
        "cutoff": False,
        "particle_azimuth_rad": particle_azimuth_rad,
        "particle_zenith_rad": particle_zenith_rad,
        "solid_angle_thrown_sr": solid_angle_thrown_sr,
    }
    return result, debug


def draw_particle_direction_with_masked_grid(
    prng,
    azimuth_deg,
    zenith_deg,
    half_angle_deg,
    energy_GeV,
    shower_spread_half_angle_deg,
    min_num_cherenkov_photons,
    allsky_deflection,
    hemisphere_grid,
    energy_factor=None,
    max_iterations=1000 * 1000,
):
    """
    Parameters
    ----------
    prng : numpy.random.Generator
        The pseudo random number-generator to draw from.

    Returns
    -------
    results : dict
        particle_azimuth_rad : float
        particle_zenith_rad : float
        solid_angle_thrown_sr : float
            The total solid angle of all masked faces in the hemispherical grid
            which where thrown in.
    debug : dict
        Additional information about how the result was obtained.
    """
    if energy_factor is None:
        energy_factor = guess_energy_factor_from_allsky_deflection(
            allsky_deflection=allsky_deflection, energy_GeV=energy_GeV
        )

    debug = {"method_key": "masked_grid"}
    debug["version"] = __version__
    debug["parameters"] = {
        "azimuth_deg": azimuth_deg,
        "zenith_deg": zenith_deg,
        "half_angle_deg": half_angle_deg,
        "energy_GeV": energy_GeV,
        "energy_factor": energy_factor,
        "shower_spread_half_angle_deg": shower_spread_half_angle_deg,
        "min_num_cherenkov_photons": min_num_cherenkov_photons,
    }
    debug["hemisphere_grid_num_vertices"] = hemisphere_grid._init_num_vertices

    assert (
        hemisphere_grid.max_zenith_distance_deg
        == allsky_deflection.config["binning"]["direction"][
            "particle_max_zenith_distance_deg"
        ]
    )

    # prime mask with matches
    # -----------------------
    try:
        matches = allsky_deflection.query_cherenkov_ball(
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=half_angle_deg,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )
    except RuntimeError as err:
        assert "Not enough population" in err.__str__()
        result = {"cutoff": True}
        return result, debug

    if len(matches) == 0:
        result = {"cutoff": True}
        return result, debug

    # cluster
    # -------
    min_samples = max([10, int(np.sqrt(len(matches)))])
    debug["cluster_min_samples"] = min_samples
    cluster_labels = estimate_cluster_labels_for_matches(
        matches=matches,
        eps_deg=(1 / 3) * shower_spread_half_angle_deg,
        min_samples=min_samples,
    )
    debug["cluster_labels"] = cluster_labels
    matches = apply_cluster_labels(
        matches=matches, cluster_labels=cluster_labels
    )
    if len(matches) == 0:
        result = {"cutoff": True}
        return result, debug

    # grid
    # ----
    debug["query_ball"] = {
        "particle_cx_rad": matches["particle_cx_rad"],
        "particle_cy_rad": matches["particle_cy_rad"],
    }

    hemisphere_mask = hemisphere.Mask(grid=hemisphere_grid)
    for i in range(len(matches)):
        hemisphere_mask.append_cx_cy(
            cx=matches["particle_cx_rad"][i],
            cy=matches["particle_cy_rad"][i],
            half_angle_deg=shower_spread_half_angle_deg,
        )

    debug["hemisphere_mask"] = list(hemisphere_mask.faces)

    # use rejection sampling to throw direction in mask
    # -------------------------------------------------
    debug["sampling"] = {
        "particle_azimuth_rad": [],
        "particle_zenith_rad": [],
        "face_idxs": [],
    }

    iteration = 0
    hit = False
    while not hit:
        (
            particle_azimuth_rad,
            particle_zenith_rad,
        ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=np.deg2rad(
                corsika_primary.MAX_ZENITH_DEG
            ),
            max_zenith_rad=np.deg2rad(corsika_primary.MAX_ZENITH_DEG),
            max_iterations=max_iterations,
        )

        face_idx = hemisphere_grid.query_azimuth_zenith(
            azimuth_deg=np.rad2deg(particle_azimuth_rad),
            zenith_deg=np.rad2deg(particle_zenith_rad),
        )

        debug["sampling"]["particle_azimuth_rad"].append(particle_azimuth_rad)
        debug["sampling"]["particle_zenith_rad"].append(particle_zenith_rad)
        debug["sampling"]["face_idxs"].append(face_idx)

        if face_idx in hemisphere_mask.faces:
            hit = True

        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Rejection-sampling failed.")

    result = {
        "cutoff": False,
        "particle_azimuth_rad": particle_azimuth_rad,
        "particle_zenith_rad": particle_zenith_rad,
        "solid_angle_thrown_sr": hemisphere_mask.solid_angle(),
    }
    return result, debug


def plot(result, debug, path):
    if debug["method_key"] == "cone":
        plot_cone(result=result, debug=debug, path=path)
    elif debug["method_key"] == "masked_grid":
        plot_masked_grid(result=result, debug=debug, path=path)
    else:
        raise KeyError("Expected either 'cone' or 'masked_grid'.")


def _style():
    s = {}
    s["result"] = {
        "fill": svgplt.color.css("red"),
        "fill_opacity": 0.8,
        "half_angle_deg": 1.5,
    }
    s["cherenkov_field_of_view"] = {
        "stroke": svgplt.color.css("blue"),
        "stroke_width": 5,
        "fill_opacity": 0.0,
    }
    s["query_ball"] = {
        "half_angle_deg": 0.5,
    }
    s["solid_angle_thrown"] = {
        "stroke": svgplt.color.css("black"),
        "fill": svgplt.color.css("green"),
        "fill_opacity": 0.5,
        "stroke_opacity": 0.25,
    }
    return s


def _ax_add_energy_marker(ax, energy_GeV):
    svgplt.ax_add_text(
        ax=ax,
        xy=[0.7, 0.9],
        text="{: 6.3f}GeV".format(energy_GeV),
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=24,
    )


def plot_cone(result, debug, path):
    sty = _style()

    fig = svgplt.Fig(cols=1080, rows=1080)
    ax = svgplt.hemisphere.Ax(fig=fig)
    _ax_add_energy_marker(ax=ax, energy_GeV=debug["parameters"]["energy_GeV"])

    # Cherenkov-field-of-view
    # -----------------------
    ax_add_field_of_view(
        ax=ax,
        azimuth_deg=debug["parameters"]["azimuth_deg"],
        zenith_deg=debug["parameters"]["zenith_deg"],
        half_angle_deg=debug["parameters"]["half_angle_deg"],
        fn=137,
        stroke=sty["cherenkov_field_of_view"]["stroke"],
        stroke_width=sty["cherenkov_field_of_view"]["stroke_width"],
        fill_opacity=sty["cherenkov_field_of_view"]["fill_opacity"],
    )

    if not result["cutoff"]:
        # matching showers in look-up-table
        # ---------------------------------
        ax_add_marker(
            ax=ax,
            cx=debug["query_ball"]["particle_cx_rad"],
            cy=debug["query_ball"]["particle_cy_rad"],
            marker_half_angle_deg=sty["query_ball"]["half_angle_deg"],
            marker_fill=None,
            marker_fill_opacity=None,
        )

        # Solid angle thrown
        # ------------------
        if debug["cone"]["is_truncated_by_max_zenith_distance"]:
            _mesh_look = svgplt.hemisphere.init_mesh_look(
                num_faces=len(debug["cone"]["faces"]),
                stroke=sty["solid_angle_thrown"]["stroke"],
                fill=sty["solid_angle_thrown"]["fill"],
                fill_opacity=sty["solid_angle_thrown"]["fill_opacity"],
                stroke_opacity=sty["solid_angle_thrown"]["stroke_opacity"],
            )
            svgplt.hemisphere.ax_add_mesh(
                ax=ax,
                vertices=debug["cone"]["vertices"],
                faces=debug["cone"]["faces"],
                max_radius=1.0,
                **_mesh_look,
            )
        else:
            ax_add_field_of_view(
                ax=ax,
                azimuth_deg=debug["average"]["particle_azimuth_deg"],
                zenith_deg=debug["average"]["particle_zenith_deg"],
                half_angle_deg=debug["cone"]["half_angle_thrown_deg"],
                fn=137,
                stroke=svgplt.color.css("black"),
                stroke_opacity=0.25,
                fill=svgplt.color.css("green"),
                fill_opacity=0.5,
            )

        # resulting particle direction which was drawn
        # --------------------------------------------
        result_cx_cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(result["particle_azimuth_rad"]),
            zenith_deg=np.rad2deg(result["particle_zenith_rad"]),
        )
        ax_add_marker(
            ax=ax,
            cx=[result_cx_cy[0]],
            cy=[result_cx_cy[1]],
            marker_half_angle_deg=sty["result"]["half_angle_deg"],
            marker_fill=[sty["result"]["fill"]],
            marker_fill_opacity=[sty["result"]["fill_opacity"]],
        )

    svgplt.hemisphere.ax_add_grid(ax=ax)
    svgplt.fig_write(fig=fig, path=path)


def plot_masked_grid(result, debug, path, hemisphere_grid=None):
    sty = _style()

    if hemisphere_grid is None:
        hemisphere_grid = hemisphere.Grid(
            num_vertices=debug["hemisphere_grid_num_vertices"]
        )
    else:
        assert (
            hemisphere_grid._init_num_vertices
            == debug["hemisphere_grid_num_vertices"]
        )

    fig = svgplt.Fig(cols=1080, rows=1080)
    ax = svgplt.hemisphere.Ax(fig=fig)
    _ax_add_energy_marker(ax=ax, energy_GeV=debug["parameters"]["energy_GeV"])

    # Cherenkov-field-of-view
    # -----------------------
    ax_add_field_of_view(
        ax=ax,
        azimuth_deg=debug["parameters"]["azimuth_deg"],
        zenith_deg=debug["parameters"]["zenith_deg"],
        half_angle_deg=debug["parameters"]["half_angle_deg"],
        fn=137,
        stroke=sty["cherenkov_field_of_view"]["stroke"],
        stroke_width=sty["cherenkov_field_of_view"]["stroke_width"],
        fill_opacity=sty["cherenkov_field_of_view"]["fill_opacity"],
    )

    if not result["cutoff"]:
        # solid angle thrown
        # ------------------
        _mesh_look = svgplt.hemisphere.init_mesh_look(
            num_faces=len(hemisphere_grid.faces),
            stroke=sty["solid_angle_thrown"]["stroke"],
            fill=sty["solid_angle_thrown"]["fill"],
            fill_opacity=0.0,
            stroke_opacity=0.05,
        )
        for i in range(len(hemisphere_grid.faces)):
            if i in debug["hemisphere_mask"]:
                _mesh_look["faces_fill"][i] = svgplt.color.css("green")
                _mesh_look["faces_fill_opacity"][i] = sty[
                    "solid_angle_thrown"
                ]["fill_opacity"]
                _mesh_look["faces_stroke_opacity"][i] = sty[
                    "solid_angle_thrown"
                ]["stroke_opacity"]
        svgplt.hemisphere.ax_add_mesh(
            ax=ax,
            vertices=hemisphere_grid.vertices,
            faces=hemisphere_grid.faces,
            max_radius=1.0,
            **_mesh_look,
        )

        # matching showers in look-up-table
        # ---------------------------------
        ax_add_marker(
            ax=ax,
            cx=debug["query_ball"]["particle_cx_rad"],
            cy=debug["query_ball"]["particle_cy_rad"],
            marker_half_angle_deg=sty["query_ball"]["half_angle_deg"],
            marker_fill=None,
            marker_fill_opacity=None,
        )

        # rejected
        # --------
        rejected_cx_cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(debug["sampling"]["particle_azimuth_rad"]),
            zenith_deg=np.rad2deg(debug["sampling"]["particle_zenith_rad"]),
        )
        ax_add_marker(
            ax=ax,
            cx=rejected_cx_cy[0],
            cy=rejected_cx_cy[1],
            marker_half_angle_deg=1.0,
            marker_fill=[
                svgplt.color.css("indigo")
                for i in range(len(rejected_cx_cy[0]))
            ],
            marker_fill_opacity=[0.25 for i in range(len(rejected_cx_cy[0]))],
        )

        # resulting particle direction which was drawn
        # --------------------------------------------
        result_cx_cy = spherical_coordinates._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(result["particle_azimuth_rad"]),
            zenith_deg=np.rad2deg(result["particle_zenith_rad"]),
        )
        ax_add_marker(
            ax=ax,
            cx=[result_cx_cy[0]],
            cy=[result_cx_cy[1]],
            marker_half_angle_deg=sty["result"]["half_angle_deg"],
            marker_fill=[sty["result"]["fill"]],
            marker_fill_opacity=[sty["result"]["fill_opacity"]],
        )

    svgplt.hemisphere.ax_add_grid(ax=ax)
    svgplt.fig_write(fig=fig, path=path)


def ax_add_marker(
    ax,
    cx,
    cy,
    marker_half_angle_deg,
    marker_fill=None,
    marker_fill_opacity=None,
    mount="altitude_azimuth_mount",
):
    assert len(cx) == len(cy)
    assert marker_half_angle_deg > 0.0

    if marker_fill is None:
        marker_fill = [svgplt.color.css("blue") for i in range(len(cx))]
    if marker_fill_opacity is None:
        marker_fill_opacity = np.ones(len(cx))

    marker_verts_uxyz = viewcone.make_ring(
        half_angle_deg=marker_half_angle_deg,
        endpoint=False,
        fn=3,
    )

    for i in range(len(cx)):
        (
            azimuth_deg,
            zenith_deg,
        ) = spherical_coordinates._cx_cy_to_az_zd_deg(cx[i], cy[i])

        rot_marker_verts_uxyz = viewcone.rotate(
            vertices_uxyz=marker_verts_uxyz,
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            mount=mount,
        )

        svgplt.ax_add_path(
            ax=ax,
            xy=rot_marker_verts_uxyz[:, 0:2],
            stroke=None,
            fill=marker_fill[i],
            fill_opacity=marker_fill_opacity[i],
        )


def ax_add_field_of_view(
    ax, azimuth_deg, zenith_deg, half_angle_deg, fn=137, **kwargs
):
    fov_ring_verts_uxyz = viewcone.make_ring(
        half_angle_deg=half_angle_deg,
        endpoint=True,
        fn=fn,
    )
    fov_ring_verts_uxyz = viewcone.rotate(
        vertices_uxyz=fov_ring_verts_uxyz,
        azimuth_deg=azimuth_deg,
        zenith_deg=zenith_deg,
    )
    svgplt.ax_add_path(ax=ax, xy=fov_ring_verts_uxyz[:, 0:2], **kwargs)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- numpy arrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))
