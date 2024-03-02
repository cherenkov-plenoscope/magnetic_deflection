from .. import utils

import un_bound_histogram
import spherical_coordinates
import spherical_histogram
import corsika_primary as cpw

import numpy as np


def report_dtype():
    return [
        # size
        # ----
        ("cherenkov_num_photons", "f4"),
        ("cherenkov_num_bunches", "u4"),
        # altitude
        # --------
        ("cherenkov_altitude_p16_m", "f4"),
        ("cherenkov_altitude_p50_m", "f4"),
        ("cherenkov_altitude_p84_m", "f4"),
        # ground
        # ------
        ("cherenkov_x_p16_m", "f4"),
        ("cherenkov_x_p50_m", "f4"),
        ("cherenkov_x_p84_m", "f4"),
        ("cherenkov_y_p16_m", "f4"),
        ("cherenkov_y_p50_m", "f4"),
        ("cherenkov_y_p84_m", "f4"),
        ("cherenkov_x_modus_m", "f4"),
        ("cherenkov_y_modus_m", "f4"),
        ("cherenkov_containment_area_p16_m2", "f4"),
        ("cherenkov_containment_area_p50_m2", "f4"),
        ("cherenkov_containment_area_p84_m2", "f4"),
        # sky
        # ---
        ("cherenkov_cx_p16", "f4"),
        ("cherenkov_cx_p50", "f4"),
        ("cherenkov_cx_p84", "f4"),
        ("cherenkov_cy_p16", "f4"),
        ("cherenkov_cy_p50", "f4"),
        ("cherenkov_cy_p84", "f4"),
        ("cherenkov_cx_modus", "f4"),
        ("cherenkov_cy_modus", "f4"),
        ("cherenkov_containment_solid_angle_p16_sr", "f4"),
        ("cherenkov_containment_solid_angle_p50_sr", "f4"),
        ("cherenkov_containment_solid_angle_p84_sr", "f4"),
        # time
        # ----
        ("cherenkov_time_p16_ns", "f4"),
        ("cherenkov_time_p50_ns", "f4"),
        ("cherenkov_time_p84_ns", "f4"),
        # above threshold
        # ---------------
        ("cherenkov_sky_num_bins_above_threshold", "u4"),
        ("cherenkov_sky_solid_angle_above_threshold_sr", "f4"),
        ("cherenkov_ground_num_bins_above_threshold", "u4"),
        ("cherenkov_ground_area_above_threshold_m2", "f4"),
    ]


class CherenkovPoolHistogram:
    def __init__(
        self,
        sky_bin_geometry,
        ground_bin_width_m,
        threshold_num_photons,
        altitude_bin_width_m=10.0,
        time_bin_duration_ns=10.0,
        cx_cy_bin_width=np.deg2rad(0.5),
    ):
        self.threshold_num_photons = threshold_num_photons

        self.altitude = un_bound_histogram.UnBoundHistogram(
            bin_width=altitude_bin_width_m
        )
        self.ground = un_bound_histogram.UnBoundHistogram2d(
            x_bin_width=ground_bin_width_m,
            y_bin_width=ground_bin_width_m,
        )
        self.sky = spherical_histogram.HemisphereHistogram(
            bin_geometry=sky_bin_geometry,
        )
        self.time_ns = un_bound_histogram.UnBoundHistogram(
            bin_width=time_bin_duration_ns
        )
        self.num_bunches = 0
        self.num_photons = 0.0

        # classic
        self.x = un_bound_histogram.UnBoundHistogram(
            bin_width=ground_bin_width_m
        )
        self.y = un_bound_histogram.UnBoundHistogram(
            bin_width=ground_bin_width_m
        )
        self.cx = un_bound_histogram.UnBoundHistogram(
            bin_width=cx_cy_bin_width
        )
        self.cy = un_bound_histogram.UnBoundHistogram(
            bin_width=cx_cy_bin_width
        )

    def reset(self):
        self.altitude.reset()
        self.ground.reset()
        self.sky.reset()
        self.x.reset()
        self.y.reset()
        self.cx.reset()
        self.cy.reset()
        self.num_bunches = 0
        self.num_photons = 0.0

    def assign_bunches(self, bunches):
        ux_to_cx = spherical_coordinates.corsika.ux_to_cx
        vy_to_cy = spherical_coordinates.corsika.vy_to_cy

        cx = ux_to_cx(ux=bunches[:, cpw.I.BUNCH.UX_1])
        cy = vy_to_cy(vy=bunches[:, cpw.I.BUNCH.VY_1])

        self.sky.assign_cx_cy(cx=cx, cy=cy)
        self.cx.assign(cx)
        self.cy.assign(cy)

        x_m = cpw.CM2M * bunches[:, cpw.I.BUNCH.X_CM]
        y_m = cpw.CM2M * bunches[:, cpw.I.BUNCH.Y_CM]
        self.ground.assign(
            x=x_m,
            y=y_m,
        )
        self.x.assign(x_m)
        self.y.assign(y_m)

        self.altitude.assign(
            x=cpw.CM2M * bunches[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
        )

        self.time_ns.assign(x=bunches[:, cpw.I.BUNCH.TIME_NS])

        self.num_photons += np.sum(bunches[:, cpw.I.BUNCH.BUNCH_SIZE_1])
        self.num_bunches += bunches.shape[0]

    def sky_above_threshold(self):
        return self.sky_intensity() > self.threshold_num_photons

    def sky_intensity(self):
        return self.sky.bin_counts

    def _report_sky_above_threshold(self):
        mask = self.sky_above_threshold()
        o = {}
        o["cherenkov_sky_num_bins_above_threshold"] = np.sum(mask)
        o["cherenkov_sky_solid_angle_above_threshold_sr"] = np.sum(
            self.sky.bin_geometry.faces_solid_angles[mask]
        )
        return o

    def _report_ground_above_threshold(self):
        _, _, counts = self.ground.to_array()
        mask = counts > self.threshold_num_photons
        total_num_bins = np.sum(mask)
        total_area = (
            total_num_bins * self.ground.x_bin_width * self.ground.y_bin_width
        )
        o = {}
        o["cherenkov_ground_num_bins_above_threshold"] = np.sum(mask)
        o["cherenkov_ground_area_above_threshold_m2"] = (
            o["cherenkov_ground_num_bins_above_threshold"]
            * self.ground.x_bin_width
            * self.ground.y_bin_width
        )
        return o

    def report(self):
        if self.num_bunches == 0:
            out = self._zero_bunches_report()
        else:
            out = self._report()

        for key, dtype in report_dtype():
            assert key in out, "Missing key {:s}".format(key)
        return out

    def _zero_bunches_report(self):
        o = {}
        for key, dtype in report_dtype():
            o[key] = float("nan")

        o["cherenkov_num_photons"] = 0.0
        o["cherenkov_num_bunches"] = 0

        o["cherenkov_sky_num_bins_above_threshold"] = 0
        o["cherenkov_sky_solid_angle_above_threshold_sr"] = 0.0
        o["cherenkov_ground_num_bins_above_threshold"] = 0
        o["cherenkov_ground_area_above_threshold_m2"] = 0.0

        return o

    def _report(self):
        o = {}
        PERCENTILES = [16, 50, 84]

        # size
        o["cherenkov_num_photons"] = self.num_photons
        o["cherenkov_num_bunches"] = self.num_bunches

        # altitude
        # --------
        for p in PERCENTILES:
            key = "cherenkov_altitude_p{:02d}_m".format(p)
            o[key] = self.altitude.percentile(p)

        # time
        # ----
        for p in PERCENTILES:
            key = "cherenkov_time_p{:02d}_ns".format(p)
            o[key] = self.time_ns.percentile(p)

        # ground
        # ------
        for p in PERCENTILES:
            key = "cherenkov_x_p{:02d}_m".format(p)
            o[key] = self.x.percentile(p)
            key = "cherenkov_y_p{:02d}_m".format(p)
            o[key] = self.y.percentile(p)

        (mx, my) = self.ground.argmax()

        o["cherenkov_x_modus_m"] = mx * self.ground.x_bin_width
        o["cherenkov_y_modus_m"] = my * self.ground.y_bin_width

        bin_area_m2 = self.ground.x_bin_width * self.ground.y_bin_width
        _, _, counts = self.ground.to_array()

        for p in PERCENTILES:
            key = "cherenkov_containment_area_p{:02d}_m2".format(p)
            o[key] = bin_area_m2 * utils.estimate_num_bins_to_contain_quantile(
                counts=counts, q=(p / 100)
            )

        # sky
        # ---
        for p in PERCENTILES:
            key = "cherenkov_cx_p{:02d}".format(p)
            o[key] = self.cx.percentile(p)
            key = "cherenkov_cy_p{:02d}".format(p)
            o[key] = self.cy.percentile(p)

        bin_counts = self.sky.bin_counts
        bin_apertures = self.sky.bin_geometry.faces_solid_angles

        for p in PERCENTILES:
            key = "cherenkov_containment_solid_angle_p{:02d}_sr".format(p)
            _, o[key] = utils.estimate_num_bins_to_contain_quantile(
                counts=bin_counts, q=(p / 100), bin_apertures=bin_apertures
            )

        max_iface = np.argmax(bin_counts)
        iv1, iv2, iv3 = self.sky.bin_geometry.faces[max_iface]
        v1 = self.sky.bin_geometry.vertices[iv1]
        v2 = self.sky.bin_geometry.vertices[iv2]
        v3 = self.sky.bin_geometry.vertices[iv3]
        cx_modus = np.mean([v1[0], v2[0], v3[0]])
        cy_modus = np.mean([v1[1], v2[1], v3[1]])

        o["cherenkov_cx_modus"] = cx_modus
        o["cherenkov_cy_modus"] = cy_modus

        o.update(self._report_sky_above_threshold())
        o.update(self._report_ground_above_threshold())
        return o
