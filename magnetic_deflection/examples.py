import os
import numpy as np


PLOTTING = {
    "light_field": {
        "cherenkov_num_photons": {
            "label": "intensity",
            "unit": "1",
            "limits": [1e3, 1e7],
        },
        "cherenkov_area_m2": {
            "label": "area",
            "unit": "m$^{2}$",
            "limits": [1e5, 1e8],
        },
        "cherenkov_solid_angle_sr": {
            "label": "solid angle",
            "unit": "sr",
            "limits": [1e-4, 1e0],
        },
        "cherenkov_density_per_m2_per_sr": {
            "label": "density",
            "unit": "m$^{-2}$ sr$^{-1}$",
            "limits": [1e-3, 1e4],
        },
        "cherenkov_density_per_m2": {
            "label": "density",
            "unit": "m$^{-2}$",
            "limits": [1e-4, 1e2],
        },
        "cherenkov_density_per_sr": {
            "label": "density",
            "unit": "sr$^{-1}$",
            "limits": [1e5, 1e9],
        },
    },
    "label_unit_seperator": "$\\,/\\,$",
    "rcParams": {
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
    },
}
