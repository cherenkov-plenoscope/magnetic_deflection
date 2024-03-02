import corsika_primary as cpw
import spherical_coordinates
import spherical_histogram
import un_bound_histogram
import os
import numpy as np
import tempfile
import atmospheric_cherenkov_response as acr
import svg_cartesian_plot as svgplt

from . import analysis
from . import cherenkov_to_primary_map
from . import cherenkov_pool_histogram

from .. import allsky


def make_example_steering(num_showers=1000, particle_id=3.0):
    return make_steering(
        run_id=1337,
        site=acr.sites.init("lapalma"),
        particle_id=particle_id,
        particle_energy_start_GeV=1.0,
        particle_energy_stop_GeV=5.0,
        particle_energy_power_slope=-2,
        particle_cone_azimuth_rad=0.0,
        particle_cone_zenith_rad=0.0,
        particle_cone_opening_angle_rad=cpw.MAX_ZENITH_DISTANCE_RAD,
        num_showers=num_showers,
    )


def make_steering(
    run_id,
    site,
    particle_id,
    particle_energy_start_GeV,
    particle_energy_stop_GeV,
    particle_energy_power_slope,
    particle_cone_azimuth_rad,
    particle_cone_zenith_rad,
    particle_cone_opening_angle_rad,
    num_showers,
):
    assert run_id > 0
    i8 = np.int64
    f8 = np.float64

    prng = np.random.Generator(np.random.PCG64(seed=run_id))

    steering = {}
    steering["run"] = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle_energy_start_GeV * 0.99),
            "stop_GeV": f8(particle_energy_stop_GeV * 1.01),
        },
        "random_seed": cpw.random.seed.make_simple_seed(seed=run_id),
    }
    steering["primaries"] = []

    for airshower_id in np.arange(1, num_showers + 1):
        az, zd = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=particle_cone_azimuth_rad,
            zenith_rad=particle_cone_zenith_rad,
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=particle_cone_opening_angle_rad,
            max_iterations=1000,
        )
        phi, theta = spherical_coordinates.corsika.az_zd_to_phi_theta(
            azimuth_rad=az,
            zenith_rad=zd,
        )
        energy_GeV = cpw.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=particle_energy_start_GeV,
            upper_limit=particle_energy_stop_GeV,
            power_slope=particle_energy_power_slope,
            num_samples=1,
        )[0]
        prm = {
            "particle_id": f8(particle_id),
            "energy_GeV": f8(energy_GeV),
            "theta_rad": f8(theta),
            "phi_rad": f8(phi),
            "depth_g_per_cm2": f8(0.0),
        }
        steering["primaries"].append(prm)

    assert len(steering["primaries"]) == num_showers
    return steering


def estimate_cherenkov_pool(corsika_steering_dict):
    pools = []
    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        with cpw.CorsikaPrimary(
            steering_dict=corsika_steering_dict,
            particle_output_path=os.path.join(tmp_dir, "corsika.par.dat"),
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        ) as corsika_run:
            for event in corsika_run:
                evth, bunch_reader = event
                corsika_bunches = np.vstack([b for b in bunch_reader])

                light_field = init_light_field_from_corsika_bunches(
                    corsika_bunches=corsika_bunches
                )

                pool = {}
                pool["run"] = int(evth[cpw.I.EVTH.RUN_NUMBER])
                pool["event"] = int(evth[cpw.I.EVTH.EVENT_NUMBER])

                par_cxcycz = particle_pointing_cxcycz(evth=evth)
                pool["particle_cx_rad"] = par_cxcycz[0]
                pool["particle_cy_rad"] = par_cxcycz[1]
                pool["particle_energy_GeV"] = evth[cpw.I.EVTH.TOTAL_ENERGY_GEV]
                pool["cherenkov_num_photons"] = np.sum(light_field["size"])
                pool["cherenkov_num_bunches"] = light_field["x"].shape[0]
                pool[
                    "cherenkov_maximum_asl_m"
                ] = estimate_cherenkov_maximum_asl_m(
                    corsika_bunches=corsika_bunches
                )
                pool.update(analysis.init(light_field=light_field))
                pools.append(pool)

        return pools


def estimate_cherenkov_maximum_asl_m(corsika_bunches):
    if len(corsika_bunches) == 0:
        return float("nan")
    else:
        return cpw.CM2M * np.median(
            corsika_bunches[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
        )


def init_light_field_from_corsika_bunches(corsika_bunches):
    cb = corsika_bunches
    lf = {}
    lf["x"] = cb[:, cpw.I.BUNCH.X_CM] * cpw.CM2M  # cm to m
    lf["y"] = cb[:, cpw.I.BUNCH.Y_CM] * cpw.CM2M  # cm to m
    lf["cx"] = spherical_coordinates.corsika.ux_to_cx(
        ux=cb[:, cpw.I.BUNCH.UX_1]
    )
    lf["cy"] = spherical_coordinates.corsika.vy_to_cy(
        vy=cb[:, cpw.I.BUNCH.VY_1]
    )
    lf["t"] = cb[:, cpw.I.BUNCH.TIME_NS] * 1e-9  # ns to s
    lf["size"] = cb[:, cpw.I.BUNCH.BUNCH_SIZE_1]
    lf["wavelength"] = cb[:, cpw.I.BUNCH.WAVELENGTH_NM] * 1e-9  # nm to m
    return lf


def particle_pointing_cxcycz(evth):
    # from momentum
    # -------------
    pointing_from_momentum_cxcycz = cpw.I.EVTH.get_pointing_cxcycz(evth=evth)
    # from angles
    # -----------
    (
        pointing_from_angles_azimuth,
        pointing_from_angles_zenith,
    ) = cpw.I.EVTH.get_pointing_az_zd(evth=evth)

    # check that momentum and angles agree
    # ------------------------------------
    pointing_from_angles_cxcycz = spherical_coordinates.az_zd_to_cx_cy_cz(
        azimuth_rad=pointing_from_angles_azimuth,
        zenith_rad=pointing_from_angles_zenith,
    )

    delta_rad = spherical_coordinates.angle_between_cx_cy_cz(
        cx1=pointing_from_momentum_cxcycz[0],
        cy1=pointing_from_momentum_cxcycz[1],
        cz1=pointing_from_momentum_cxcycz[2],
        cx2=pointing_from_angles_cxcycz[0],
        cy2=pointing_from_angles_cxcycz[1],
        cz2=pointing_from_angles_cxcycz[2],
    )
    assert delta_rad < (2.0 * np.pi * 1e-4)

    return pointing_from_momentum_cxcycz


def histogram_cherenkov_pool(
    corsika_steering_dict,
    binning,
    threshold_num_photons,
):
    cer_to_prm = cherenkov_to_primary_map.CherenkovToPrimaryMap(
        sky_bin_geometry=binning["sky"],
        energy_bin_edges_GeV=binning["energy"]["edges"],
        altitude_bin_edges_m=binning["altitude"]["edges"],
    )

    reports = []
    cerpoolhist = cherenkov_pool_histogram.CherenkovPoolHistogram(
        sky_bin_geometry=binning["sky"],
        ground_bin_width_m=binning["ground"]["width"],
        threshold_num_photons=threshold_num_photons,
    )

    with tempfile.TemporaryDirectory(prefix="mdfl_") as tmp_dir:
        with cpw.CorsikaPrimary(
            steering_dict=corsika_steering_dict,
            particle_output_path=os.path.join(tmp_dir, "corsika.par.dat"),
            stdout_path=os.path.join(tmp_dir, "corsika.stdout"),
            stderr_path=os.path.join(tmp_dir, "corsika.stderr"),
        ) as corsika_run:
            for event in corsika_run:
                evth, bunch_reader = event

                cerpoolhist.reset()

                print(len(reports))

                report = {}
                report["run"] = int(evth[cpw.I.EVTH.RUN_NUMBER])
                report["event"] = int(evth[cpw.I.EVTH.EVENT_NUMBER])

                par_cxcycz = particle_pointing_cxcycz(evth=evth)
                report["particle_cx"] = par_cxcycz[0]
                report["particle_cy"] = par_cxcycz[1]
                report["particle_energy_GeV"] = evth[
                    cpw.I.EVTH.TOTAL_ENERGY_GEV
                ]

                for bunches in bunch_reader:
                    cerpoolhist.assign_bunches(bunches=bunches)

                report.update(cerpoolhist.report())
                cherenkov_sky_mask = cerpoolhist.sky_above_threshold()

                cermap_report = cer_to_prm.assign(
                    particle_cx=report["particle_cx"],
                    particle_cy=report["particle_cy"],
                    particle_energy_GeV=report["particle_energy_GeV"],
                    cherenkov_altitude_p50_m=report[
                        "cherenkov_altitude_p50_m"
                    ],
                    cherenkov_sky_mask=cherenkov_sky_mask,
                )
                report.update(cermap_report)

                reports.append(report)
        return reports, cer_to_prm


def plot_histogram_cherenkov_pool(path, pool):
    hemi_hist = spherical_histogram.HemisphereHistogram(
        num_vertices=NUM_VERTICES,
        max_zenith_distance_rad=MAX_ZENITH_DISTANCE_RAD,
    )

    fig = svgplt.Fig(cols=1080, rows=1080)
    ax = svgplt.hemisphere.Ax(fig=fig)

    svgplt.ax_add_text(
        ax=ax,
        xy=[0.0, 1.1],
        text="energy {:.2f} GeV".format(pool["particle_energy_GeV"]),
        fill=svgplt.color.css("black"),
        font_family="math",
        font_size=30,
    )

    _mesh_look = svgplt.hemisphere.init_mesh_look(
        num_faces=len(hemi_hist.bin_geometry.faces),
        stroke=svgplt.color.css("blue"),
        fill=svgplt.color.css("green"),
        fill_opacity=0.0,
        stroke_opacity=0.0,
    )

    if len(pool["hemisphere"]["bins"]) == 0:
        vmax = 1
    else:
        vmax = max(
            [
                pool["hemisphere"]["bins"][bb]
                for bb in pool["hemisphere"]["bins"]
            ]
        )
    for i in range(len(hemi_hist.bin_geometry.faces)):
        if i in pool["hemisphere"]["bins"]:
            _mesh_look["faces_fill_opacity"][i] = (
                pool["hemisphere"]["bins"][i] / vmax
            )
            _mesh_look["faces_stroke_opacity"][i] = (
                pool["hemisphere"]["bins"][i] / vmax
            )

    # particle direction
    # ------------------
    allsky.random.ax_add_marker(
        ax=ax,
        cx=[pool["particle_cx_rad"]],
        cy=[pool["particle_cy_rad"]],
        marker_half_angle_rad=np.deg2rad(1.0),
        marker_fill_all=svgplt.color.css("red"),
        marker_fill_opacity_all=0.8,
    )
    svgplt.hemisphere.ax_add_mesh(
        ax=ax,
        vertices=hemi_hist.bin_geometry.vertices,
        faces=hemi_hist.bin_geometry.faces,
        max_radius=1.0,
        **_mesh_look,
    )

    svgplt.hemisphere.ax_add_grid(ax=ax)
    svgplt.fig_write(fig=fig, path=path)
    from svg_cartesian_plot import inkscape

    inkscape.render(
        svg_path=path,
        out_path=path + ".png",
        background_opacity=0.0,
        export_type="png",
    )
