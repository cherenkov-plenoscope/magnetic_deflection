import os
import numpy as np

SITES = {
    "namibia": {
        "observation_level_asl_m": 2300,
        "earth_magnetic_field_x_muT": 12.5,
        "earth_magnetic_field_z_muT": -25.9,
        "atmosphere_id": 10,
    },
    "namibiaOff": {
        "observation_level_asl_m": 2300,
        "earth_magnetic_field_x_muT": 1e-06,
        "earth_magnetic_field_z_muT": 1e-06,
        "atmosphere_id": 10,
    },
    "chile": {
        "observation_level_asl_m": 5000,
        "earth_magnetic_field_x_muT": 20.815,
        "earth_magnetic_field_z_muT": -11.366,
        "atmosphere_id": 26,
    },
    "lapalma": {
        "observation_level_asl_m": 2200,
        "earth_magnetic_field_x_muT": 30.419,
        "earth_magnetic_field_z_muT": 23.856,
        "atmosphere_id": 8,
    },
}


PARTICLES = {
    "gamma": {
        "particle_id": 1,
        "energy_bin_edges_GeV": [0.5, 100],
        "electric_charge_qe": 0.0,
        "magnetic_deflection_max_off_axis_deg": 0.25,
    },
    "electron": {
        "particle_id": 3,
        "energy_bin_edges_GeV": [0.5, 100],
        "electric_charge_qe": -1.0,
        "magnetic_deflection_max_off_axis_deg": 0.5,
    },
    "proton": {
        "particle_id": 14,
        "energy_bin_edges_GeV": [5, 100],
        "electric_charge_qe": 1.0,
        "magnetic_deflection_max_off_axis_deg": 1.5,
    },
    "helium": {
        "particle_id": 402,
        "energy_bin_edges_GeV": [10, 100],
        "electric_charge_qe": 2.0,
        "magnetic_deflection_max_off_axis_deg": 1.5,
    },
}

POINTING = {"azimuth_deg": 0.0, "zenith_deg": 0.0}

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
    "sites": {
        "namibia": {"label": "Gamsberg", "marker": "+", "linestyle": "--",},
        "chile": {"label": "Chajnantor", "marker": "*", "linestyle": ":",},
        "namibiaOff": {
            "label": "Gamsberg-Off",
            "marker": ".",
            "linestyle": "-.",
        },
        "lapalma": {"label": "Roque", "marker": "^", "linestyle": "-",},
    },
    "particles": {
        "gamma": {"color": "black", "label": "gamma-ray",},
        "electron": {"color": "blue", "label": "electron",},
        "proton": {"color": "red", "label": "proton",},
        "helium": {"color": "orange", "label": "helium",},
    },
    "light_field": {
        "num_photons": {
            "label": "intensity",
            "unit": "1",
            "limits": [1e2, 1e6],
        },
        "spread_area_m2": {
            "label": "area",
            "unit": "m$^{2}$",
            "limits": [1e4, 1e7],
        },
        "spread_solid_angle_deg2": {
            "label": "solid angle",
            "unit": "$(1^\\circ)^{2}$",
            "limits": [1e-1, 1e2],
        },
        "light_field_outer_density": {
            "label": "density",
            "unit": "m$^{-2} (1^\\circ)^{-2}$",
            "limits": [1e-3, 1e2],
        },
    },
    "label_unit_seperator": "$\\,/\\,$",
    "rcParams": {"mathtext.fontset": "cm", "font.family": "STIXGeneral",},
}

TYPICAL_NSB_IN_PMT_PER_M2_PER_SR_PER_S = 5e11

CM_OPENING_ANGLE_DEG = 0.05
CM_OPENING_ANGLE_RAD = np.deg2rad(CM_OPENING_ANGLE_DEG)
CM_APERTURE_RADIUS_M = 35.5
CM_APERTURE_AREA_M2 = np.pi *CM_APERTURE_RADIUS_M ** 2
CM_SOLID_ANGLE_SR = np.pi * CM_OPENING_ANGLE_RAD ** 2
CM_ACCEPTANCE_M2_SR = CM_APERTURE_AREA_M2 * CM_SOLID_ANGLE_SR

CM_NSB_RATE = TYPICAL_NSB_IN_PMT_PER_M2_PER_SR_PER_S * CM_ACCEPTANCE_M2_SR
CM_NSB_INTENSITY_IN_1NS = CM_NSB_RATE / 1e9

DENSITY_CUT_MEDIAN = {"median": {"percentile": 50.0},}
DENSITY_CUT_NUM_NEIGHBORS = {
    "num_neighbors": {
        "xy_radius": CM_APERTURE_RADIUS_M,
        "cxcy_radius": CM_OPENING_ANGLE_RAD,
        "min_num_neighbors": int(CM_NSB_INTENSITY_IN_1NS)
    }
}
