import numpy as np
import corsika_primary
import svg_cartesian_plot as svgplt
import solid_angle_utils

from . import hemisphere
from . import viewcone
import spherical_coordinates
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


def guess_min_samples_for_clustering(num_showers, min_samples=10):
    return max([min_samples, int(np.sqrt(num_showers))])


def guess_eps_rad_for_clustering(shower_spread_half_angle_rad):
    arbitrary_factor = 2.5
    return arbitrary_factor * np.sqrt(shower_spread_half_angle_rad)


def estimate_cluster_labels_for_showers(showers, eps_rad, min_samples):
    _cz = spherical_coordinates.restore_cz(
        cx=showers["particle_cx_rad"], cy=showers["particle_cy_rad"]
    )
    particle_cxcycz = np.c_[
        showers["particle_cx_rad"], showers["particle_cy_rad"], _cz
    ]
    labels = hemisphere.cluster(
        vertices=particle_cxcycz,
        eps=eps_rad,
        min_samples=min_samples,
    )
    return labels


def cluster_showers_to_remove_outliers(
    showers,
    shower_spread_half_angle_rad,
):
    dbg = {}
    dbg["min_samples"] = guess_min_samples_for_clustering(
        num_showers=len(showers)
    )
    dbg["eps_rad"] = guess_eps_rad_for_clustering(
        shower_spread_half_angle_rad=shower_spread_half_angle_rad
    )
    dbg["labels"] = estimate_cluster_labels_for_showers(
        showers=showers,
        eps_rad=dbg["eps_rad"],
        min_samples=dbg["min_samples"],
    )
    dbg["mask"] = dbg["labels"] >= 0  # everything but outliers

    dense_showers = showers[dbg["mask"]]
    return dense_showers, dbg


def draw_particle_direction_with_cone(
    prng,
    azimuth_rad,
    zenith_rad,
    half_angle_rad,
    energy_GeV,
    shower_spread_half_angle_rad,
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
        "azimuth_rad": azimuth_rad,
        "zenith_rad": zenith_rad,
        "half_angle_rad": half_angle_rad,
        "energy_GeV": energy_GeV,
        "energy_factor": energy_factor,
        "shower_spread_half_angle_rad": shower_spread_half_angle_rad,
        "min_num_cherenkov_photons": min_num_cherenkov_photons,
    }

    max_zenith_distance_rad = allsky_deflection.config["binning"]["direction"][
        "particle_max_zenith_distance_rad"
    ]
    debug["max_zenith_distance_rad"] = max_zenith_distance_rad

    try:
        (
            showers,
            direction_weights,
            energy_weights,
        ) = allsky_deflection.query_cherenkov_ball(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
            weights=True,
        )
    except RuntimeError as err:
        assert "Not enough population" in err.__str__()
        result = {"cutoff": True}
        return result, debug

    debug["query_ball"] = {
        "particle_cx_rad": showers["particle_cx_rad"],
        "particle_cy_rad": showers["particle_cy_rad"],
        "energy_weights": energy_weights,
    }

    if len(showers) == 0:
        result = {"cutoff": True}
        return result, debug

    # cluster
    # -------
    dense_showers, cluster_debug = cluster_showers_to_remove_outliers(
        showers=showers,
        shower_spread_half_angle_rad=shower_spread_half_angle_rad,
    )
    dense_energy_weights = energy_weights[cluster_debug["mask"]]
    debug["cluster"] = cluster_debug
    debug["cluster"]["particle_cx_rad"] = dense_showers["particle_cx_rad"]
    debug["cluster"]["particle_cy_rad"] = dense_showers["particle_cy_rad"]
    debug["cluster"]["energy_weights"] = dense_energy_weights

    if len(dense_showers) == 0:
        result = {"cutoff": True}
        return result, debug

    # cone
    # ----
    avg_particle_cx_rad, std_particle_cx_rad = weighted_avg_and_std(
        values=dense_showers["particle_cx_rad"], weights=dense_energy_weights
    )
    avg_particle_cy_rad, std_particle_cy_rad = weighted_avg_and_std(
        values=dense_showers["particle_cy_rad"], weights=dense_energy_weights
    )

    (
        avg_particle_azimuth_rad,
        avg_particle_zenith_rad,
    ) = spherical_coordinates.cx_cy_to_az_zd(
        cx=avg_particle_cx_rad, cy=avg_particle_cy_rad
    )

    debug["average"] = {
        "particle_azimuth_rad": avg_particle_azimuth_rad,
        "particle_zenith_rad": avg_particle_zenith_rad,
    }

    assert avg_particle_zenith_rad <= max_zenith_distance_rad

    half_angle_thrown_rad = shower_spread_half_angle_rad + half_angle_rad

    debug["cone"] = {}
    debug["cone"]["half_angle_thrown_rad"] = half_angle_thrown_rad

    if (
        avg_particle_zenith_rad + half_angle_thrown_rad
        <= max_zenith_distance_rad
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
            half_angle_rad=half_angle_thrown_rad,
            endpoint=True,
            fn=137,
        )
        cone_vertices = viewcone.rotate(
            vertices_uxyz=cone_vertices,
            azimuth_rad=avg_particle_azimuth_rad,
            zenith_rad=avg_particle_zenith_rad,
        )
        cone_vertices = viewcone.limit_zenith_distance(
            vertices_uxyz=cone_vertices,
            max_zenith_distance_rad=max_zenith_distance_rad,
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
        azimuth_rad=avg_particle_azimuth_rad,
        zenith_rad=avg_particle_zenith_rad,
        min_scatter_opening_angle_rad=0.0,
        max_scatter_opening_angle_rad=half_angle_thrown_rad,
        max_zenith_rad=corsika_primary.MAX_ZENITH_RAD,
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
    azimuth_rad,
    zenith_rad,
    half_angle_rad,
    energy_GeV,
    shower_spread_half_angle_rad,
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
        "azimuth_rad": azimuth_rad,
        "zenith_rad": zenith_rad,
        "half_angle_rad": half_angle_rad,
        "energy_GeV": energy_GeV,
        "energy_factor": energy_factor,
        "shower_spread_half_angle_rad": shower_spread_half_angle_rad,
        "min_num_cherenkov_photons": min_num_cherenkov_photons,
    }
    debug["hemisphere_grid_num_vertices"] = hemisphere_grid._init_num_vertices

    assert (
        hemisphere_grid.max_zenith_distance_rad
        == allsky_deflection.config["binning"]["direction"][
            "particle_max_zenith_distance_rad"
        ]
    )

    # prime mask with showers
    # -----------------------
    try:
        showers = allsky_deflection.query_cherenkov_ball(
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
            half_angle_rad=half_angle_rad,
            energy_GeV=energy_GeV,
            energy_factor=energy_factor,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )
    except RuntimeError as err:
        assert "Not enough population" in err.__str__()
        result = {"cutoff": True}
        return result, debug

    debug["query_ball"] = {
        "particle_cx_rad": showers["particle_cx_rad"],
        "particle_cy_rad": showers["particle_cy_rad"],
    }

    if len(showers) == 0:
        result = {"cutoff": True}
        return result, debug

    # cluster
    # -------
    dense_showers, cluster_debug = cluster_showers_to_remove_outliers(
        showers=showers,
        shower_spread_half_angle_rad=shower_spread_half_angle_rad,
    )
    debug["cluster"] = cluster_debug
    debug["cluster"]["particle_cx_rad"] = dense_showers["particle_cx_rad"]
    debug["cluster"]["particle_cy_rad"] = dense_showers["particle_cy_rad"]

    if len(dense_showers) == 0:
        result = {"cutoff": True}
        return result, debug

    # grid
    # ----
    hemisphere_mask = hemisphere.Mask(grid=hemisphere_grid)
    for i in range(len(dense_showers)):
        hemisphere_mask.append_cx_cy(
            cx=dense_showers["particle_cx_rad"][i],
            cy=dense_showers["particle_cy_rad"][i],
            half_angle_rad=shower_spread_half_angle_rad,
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
            max_scatter_opening_angle_rad=corsika_primary.MAX_ZENITH_RAD,
            max_zenith_rad=corsika_primary.MAX_ZENITH_RAD,
            max_iterations=max_iterations,
        )

        face_idx = hemisphere_grid.query_azimuth_zenith(
            azimuth_rad=particle_azimuth_rad,
            zenith_rad=particle_zenith_rad,
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


class Random:
    """
    Draw the direction of primary particles.
    """

    def __init__(self, allsky_deflection, hemisphere_grid=None):
        self.allsky_deflection = allsky_deflection
        self.hemisphere_grid = hemisphere_grid

    def _make_hemisphere_grid_if_needed(self):
        if self.hemisphere_grid is None:
            self.hemisphere_grid = hemisphere.Grid(num_vertices=4096)

    def draw_particle_direction(
        self,
        prng,
        method,
        azimuth_rad,
        zenith_rad,
        half_angle_rad,
        energy_GeV,
        shower_spread_half_angle_rad,
        min_num_cherenkov_photons=1e3,
    ):
        """
        Query, and draw the direction of a primary particle to induce an
        atmospheric shower which emitts its light (median) in a specific
        direction.

        Parameters
        ----------
        prng : numpy.random.Generator()
            Pseudo random number generator.
        method : str
            Either 'cone' or 'grid'.
        azimuth_rad : float
            Cherenkov light's azimuth.
        zenith_rad : float
            Cherenkov light's zenith.
        half_angle_rad : float
            Field-of-view for Cherenkov light.
        energy_GeV : float
            Primary particle's energy.
        shower_spread_half_angle_rad : float
            Spread to widen the solid angle from where the primarie's direction
            is drawn.
        min_num_cherenkov_photons : float
            When querying the database, only use showers with this many Cherenkov
            photons.
        hemisphere_grid : magnetic_deflection.allsky.hemisphere.Grid()
            Only relevant for method 'grid'. When drawing multiple times, provide
            a grid here.

        Returns
        -------
        result, debug : (dict, dict)
            result = {
                cutoff : bool
                particle_azimuth_rad : float
                particle_zenith_rad : float
                solid_angle_thrown_sr : float
            }
            If cutoff is True, there is no valid trajectory of primary particles
            to create Cherenkov light in the queried direction.
                The debug contains details about the query and the drawing.
        """
        if method == "cone":
            (res, dbg) = draw_particle_direction_with_cone(
                prng=prng,
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                half_angle_rad=half_angle_rad,
                energy_GeV=energy_GeV,
                shower_spread_half_angle_rad=shower_spread_half_angle_rad,
                min_num_cherenkov_photons=min_num_cherenkov_photons,
                allsky_deflection=self.allsky_deflection,
            )
        elif method == "grid":
            self._make_hemisphere_grid_if_needed()

            (res, dbg) = draw_particle_direction_with_masked_grid(
                prng=prng,
                azimuth_rad=azimuth_rad,
                zenith_rad=zenith_rad,
                half_angle_rad=half_angle_rad,
                energy_GeV=energy_GeV,
                shower_spread_half_angle_rad=shower_spread_half_angle_rad,
                min_num_cherenkov_photons=min_num_cherenkov_photons,
                allsky_deflection=self.allsky_deflection,
                hemisphere_grid=self.hemisphere_grid,
            )
        else:
            raise KeyError("Expected either 'grid' or 'cone'.")

        return res, dbg


def _style():
    s = {}
    s["result"] = {
        "fill": svgplt.color.css("red"),
        "fill_opacity": 0.8,
        "half_angle_rad": np.deg2rad(1.5),
    }
    s["cherenkov_field_of_view"] = {
        "stroke": svgplt.color.css("blue"),
        "stroke_width": 5,
        "fill_opacity": 0.0,
    }
    s["query_ball"] = {
        "half_angle_rad": np.deg2rad(0.5),
        "fill": svgplt.color.css("gray"),
        "fill_opacity": 0.25,
    }
    s["cluster"] = {
        "half_angle_rad": np.deg2rad(0.5),
        "fill": svgplt.color.css("blue"),
        "fill_opacity": 1,
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


def _ax_add_cutoff_marker(ax):
    svgplt.ax_add_text(
        ax=ax,
        xy=[0.0, 0.0],
        text="cutoff",
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
        azimuth_rad=debug["parameters"]["azimuth_rad"],
        zenith_rad=debug["parameters"]["zenith_rad"],
        half_angle_rad=debug["parameters"]["half_angle_rad"],
        fn=137,
        stroke=sty["cherenkov_field_of_view"]["stroke"],
        stroke_width=sty["cherenkov_field_of_view"]["stroke_width"],
        fill_opacity=sty["cherenkov_field_of_view"]["fill_opacity"],
    )

    if not result["cutoff"]:
        # Solid angle thrown
        # ------------------
        if debug["cone"]["is_truncated_by_max_zenith_distance"]:
            _mesh_look = svgplt.hemisphere.init_mesh_look(
                num_faces=len(debug["cone"]["faces"]),
                stroke=sty["solid_angle_thrown"]["stroke"],
                stroke_opacity=sty["solid_angle_thrown"]["stroke_opacity"],
                fill=sty["solid_angle_thrown"]["fill"],
                fill_opacity=sty["solid_angle_thrown"]["fill_opacity"],
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
                azimuth_rad=debug["average"]["particle_azimuth_rad"],
                zenith_rad=debug["average"]["particle_zenith_rad"],
                half_angle_rad=debug["cone"]["half_angle_thrown_rad"],
                fn=137,
                stroke=sty["solid_angle_thrown"]["stroke"],
                stroke_opacity=sty["solid_angle_thrown"]["stroke_opacity"],
                fill=sty["solid_angle_thrown"]["fill"],
                fill_opacity=sty["solid_angle_thrown"]["fill_opacity"],
            )

        # showers in query_ball
        # ---------------------
        ax_add_marker(
            ax=ax,
            cx=debug["query_ball"]["particle_cx_rad"],
            cy=debug["query_ball"]["particle_cy_rad"],
            marker_half_angle_rad=sty["query_ball"]["half_angle_rad"],
            marker_fill_all=sty["query_ball"]["fill"],
            marker_fill_opacity_all=sty["query_ball"]["fill_opacity"],
        )

        # showers in cluster
        # ------------------
        ax_add_marker(
            ax=ax,
            cx=debug["cluster"]["particle_cx_rad"],
            cy=debug["cluster"]["particle_cy_rad"],
            marker_half_angle_rad=sty["cluster"]["half_angle_rad"],
            marker_fill_all=sty["cluster"]["fill"],
            marker_fill_opacity_all=sty["cluster"]["fill_opacity"],
        )

        # resulting particle direction which was drawn
        # --------------------------------------------
        result_cx_cy = spherical_coordinates.az_zd_to_cx_cy(
            azimuth_rad=np.rad2deg(result["particle_azimuth_rad"]),
            zenith_rad=np.rad2deg(result["particle_zenith_rad"]),
        )
        ax_add_marker(
            ax=ax,
            cx=[result_cx_cy[0]],
            cy=[result_cx_cy[1]],
            marker_half_angle_rad=sty["result"]["half_angle_rad"],
            marker_fill_all=sty["result"]["fill"],
            marker_fill_opacity_all=sty["result"]["fill_opacity"],
        )
    else:
        _ax_add_cutoff_marker(ax=ax)

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
        azimuth_rad=debug["parameters"]["azimuth_rad"],
        zenith_rad=debug["parameters"]["zenith_rad"],
        half_angle_rad=debug["parameters"]["half_angle_rad"],
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
            stroke_opacity=0.0,
        )
        for i in range(len(hemisphere_grid.faces)):
            if i in debug["hemisphere_mask"]:
                _mesh_look["faces_fill"][i] = sty["solid_angle_thrown"]["fill"]
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

        # showers in query_ball
        # ---------------------
        ax_add_marker(
            ax=ax,
            cx=debug["query_ball"]["particle_cx_rad"],
            cy=debug["query_ball"]["particle_cy_rad"],
            marker_half_angle_rad=sty["query_ball"]["half_angle_rad"],
            marker_fill_all=sty["query_ball"]["fill"],
            marker_fill_opacity_all=sty["query_ball"]["fill_opacity"],
        )

        # showers in cluster
        # ------------------
        ax_add_marker(
            ax=ax,
            cx=debug["cluster"]["particle_cx_rad"],
            cy=debug["cluster"]["particle_cy_rad"],
            marker_half_angle_rad=sty["cluster"]["half_angle_rad"],
            marker_fill_all=sty["cluster"]["fill"],
            marker_fill_opacity_all=sty["cluster"]["fill_opacity"],
        )

        # rejected
        # --------
        if False:
            rejected_cx_cy = spherical_coordinates.az_zd_to_cx_cy(
                azimuth_rad=debug["sampling"]["particle_azimuth_rad"],
                zenith_rad=debug["sampling"]["particle_zenith_rad"],
            )
            ax_add_marker(
                ax=ax,
                cx=rejected_cx_cy[0],
                cy=rejected_cx_cy[1],
                marker_half_angle_rad=np.deg2rad(1.0),
                marker_fill_all=svgplt.color.css("indigo"),
                marker_fill_opacity_all=0.25,
            )

        # resulting particle direction which was drawn
        # --------------------------------------------
        result_cx_cy = spherical_coordinates.az_zd_to_cx_cy(
            azimuth_rad=result["particle_azimuth_rad"],
            zenith_rad=result["particle_zenith_rad"],
        )
        ax_add_marker(
            ax=ax,
            cx=[result_cx_cy[0]],
            cy=[result_cx_cy[1]],
            marker_half_angle_rad=sty["result"]["half_angle_rad"],
            marker_fill_all=sty["result"]["fill"],
            marker_fill_opacity_all=sty["result"]["fill_opacity"],
        )
    else:
        _ax_add_cutoff_marker(ax=ax)

    svgplt.hemisphere.ax_add_grid(ax=ax)
    svgplt.fig_write(fig=fig, path=path)


def ax_add_marker(
    ax,
    cx,
    cy,
    marker_half_angle_rad,
    marker_fill_all=None,
    marker_fill_opacity_all=None,
    mount="altitude_azimuth_mount",
):
    assert len(cx) == len(cy)
    assert marker_half_angle_rad > 0.0

    if marker_fill_all is not None:
        marker_fill = [marker_fill_all for i in range(len(cx))]
    if marker_fill_opacity_all is not None:
        marker_fill_opacity = [marker_fill_opacity_all for i in range(len(cx))]

    marker_verts_uxyz = viewcone.make_ring(
        half_angle_rad=marker_half_angle_rad,
        endpoint=False,
        fn=3,
    )

    for i in range(len(cx)):
        azimuth_rad, zenith_rad = spherical_coordinates.cx_cy_to_az_zd(
            cx[i], cy[i]
        )

        rot_marker_verts_uxyz = viewcone.rotate(
            vertices_uxyz=marker_verts_uxyz,
            azimuth_rad=azimuth_rad,
            zenith_rad=zenith_rad,
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
    ax, azimuth_rad, zenith_rad, half_angle_rad, fn=137, **kwargs
):
    fov_ring_verts_uxyz = viewcone.make_ring(
        half_angle_rad=half_angle_rad,
        endpoint=True,
        fn=fn,
    )
    fov_ring_verts_uxyz = viewcone.rotate(
        vertices_uxyz=fov_ring_verts_uxyz,
        azimuth_rad=azimuth_rad,
        zenith_rad=zenith_rad,
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
