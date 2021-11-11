import os

SITE_CHILE = {
    "earth_magnetic_field_x_muT": 20.815,
    "earth_magnetic_field_z_muT": -11.366,
    "observation_level_asl_m": 5e3,
    "atmosphere_id": 26,
}

SITE_NAMIBIA = {
    "earth_magnetic_field_x_muT": 12.5,
    "earth_magnetic_field_z_muT": -25.9,
    "observation_level_asl_m": 2300,
    "atmosphere_id": 10,
}

PARTICLE_ELECTRON = {
    "particle_id": 3,
    "energy_bin_edges_GeV": [0.5, 100],
    "max_scatter_angle_deg": 10,
    "energy_power_law_slope": -1.7,
    "electric_charge_qe": -1.0,
    "magnetic_deflection_max_off_axis_deg": 0.5,
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
        "namibia": {
            "label": "Gamsberg",
            "marker": "+",
            "linestyle": "--",
        },
        "chile": {
            "label": "Chajnantor",
            "marker": "*",
            "linestyle": ":",
        },
        "namibiaOff": {
            "label": "Gamsberg-Off",
            "marker": ".",
            "linestyle": "-.",
        },
        "lapalma": {
            "label": "Roque",
            "marker": "^",
            "linestyle": "-",
        },
    },
    "particles": {
        "gamma": {
            "color": "black",
            "label": "gamma-ray",
        },
        "electron": {
            "color": "blue",
            "label": "electron",
        },
        "proton": {
            "color": "red",
            "label": "proton",
        },
        "helium": {
            "color": "orange",
            "label": "helium",
        },
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
        "density": {
            "label": "density",
            "unit": "m$^{-2}\\,(1^\\circ)^{-2}$",
            "limits": [1e-3, 1e2],
        },
    },
}
