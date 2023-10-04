import os
import numpy as np

CORSIKA_PRIMARY_MOD_PATH = os.path.abspath(
    os.path.join(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd",
    )
)

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

TYPICAL_NSB_IN_PMT_PER_M2_PER_SR_PER_S = 5e11

CM_OPENING_ANGLE_DEG = 0.05
CM_OPENING_ANGLE_RAD = np.deg2rad(CM_OPENING_ANGLE_DEG)
CM_APERTURE_RADIUS_M = 35.5
CM_APERTURE_AREA_M2 = np.pi * CM_APERTURE_RADIUS_M**2
CM_SOLID_ANGLE_SR = np.pi * CM_OPENING_ANGLE_RAD**2
CM_ACCEPTANCE_M2_SR = CM_APERTURE_AREA_M2 * CM_SOLID_ANGLE_SR

CM_NSB_RATE = TYPICAL_NSB_IN_PMT_PER_M2_PER_SR_PER_S * CM_ACCEPTANCE_M2_SR
CM_NSB_INTENSITY_IN_1NS = CM_NSB_RATE / 1e9
