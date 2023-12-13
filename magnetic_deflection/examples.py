import os
import numpy as np
import binning_utils


def hemisphere_field_of_view():
    return {
        "wide": {
            "angle_deg": 90,
            "zenith_mayor_deg": [],
            "zenith_minor_deg": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        },
        "narrow": {
            "angle_deg": 10,
            "zenith_mayor_deg": [],
            "zenith_minor_deg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    }


def common_energy_limits():
    return {"energy_start_GeV": 0.1, "energy_stop_GeV": 100}
