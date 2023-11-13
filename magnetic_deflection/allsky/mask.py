import scipy
from .. import spherical_coordinates
from scipy import spatial
import solid_angle_utils
import binning_utils
import numpy as np
import svg_cartesian_plot as splt
import copy


class Mask:
    def __init__(self, num_vertices=1024, max_zenith_distance_deg=90):
        assert num_vertices >= 1

        self.num_vertices = int(num_vertices)
        self.max_zenith_distance_deg = int(max_zenith_distance_deg)
        self.vertices = binning_utils.sphere.fibonacci_space(
            size=self.num_vertices,
            max_zenith_distance_rad=np.deg2rad(self.max_zenith_distance_deg),
        )
