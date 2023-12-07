import magnetic_deflection as mdfl
import os
import binning_utils

import sebastians_matplotlib_addons as sebplt
import matplotlib
from matplotlib.colors import ListedColormap

PLT = mdfl.examples.PLOTTING
sebplt.matplotlib.rcParams.update(PLT["rcParams"])
FIGSIZE = {"rows": 720, "cols": 1280, "fontsize": 1.25}


PCOLORS = {
    "gamma": "black",
    "electron": "blue",
    "proton": "red",
    "helium": "orange",
}

energy_bin = binning_utils.Binning(bin_edges=np.geomspace(1e-1, 1e2, 65))

out_dir = "magsky_plots"
work_dir = "magsky"
azimuth_deg = 180.0
zenith_deg = 0.0
half_angle_deg = 6.5

os.makedirs(out_dir, exist_ok=True)

site_keys, particle_keys = mdfl.production.find_site_and_particle_keys(
    work_dir=work_dir
)


res = {}

for sk in site_keys:
    res[sk] = {}
    for pk in particle_keys:
        res[sk][pk] = {}

        print("load", sk, pk)

        allsky = mdfl.allsky.AllSky(
            work_dir=os.path.join(work_dir, sk, pk),
            cache_dtype=mdfl.allsky.store.page.dtype(),
        )

        showers = mdfl.allsky.analysis.query_cherenkov_ball_in_all_energy(
            allsky=allsky,
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            half_angle_deg=half_angle_deg,
            min_num_cherenkov_photons=1e3,
        )

        res[sk][pk]["particle_energy_GeV"] = showers["particle_energy_GeV"]
        res[sk][pk]["cherenkov_num_photons"] = showers["cherenkov_num_photons"]
        for percentile in [50, 90]:
            dens = mdfl.allsky.analysis.estimate_cherenkov_density(
                showers=showers,
                percentile=percentile,
            )
            res[sk][pk].update(dens)

        del showers
        del allsky
        del dens


def make_cmap(color):
    o = np.ones(256)
    alpha = np.linspace(0, 1, 256)
    mycolors = np.c_[o * color[0], o * color[1], o * color[2], alpha]
    cmap = ListedColormap(mycolors)
    return cmap


PCOLORMAPS = {
    "gamma": make_cmap([0, 0, 0]),
    "electron": make_cmap([0, 0, 1]),
    "proton": make_cmap([1, 0, 0]),
    "helium": make_cmap([1, 0.65, 0]),
}

DENSITIES = {
    "cherenkov_area_density50_per_m2": {
        "label": "area density",
        "unit": r"m$^{-2}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-8, 1e1, 33)),
    },
    "cherenkov_light_field_density50_per_m2_per_sr": {
        "label": "light-field density",
        "unit": r"m$^{-2}$ sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-5, 1e5, 33)),
    },
    "cherenkov_solid_angle_density50_per_sr": {
        "label": "solid angle density",
        "unit": r"sr$^{-1}$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e1, 1e12, 33)),
    },
    "cherenkov_num_photons": {
        "label": "size",
        "unit": r"1",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e3, 1e8, 33)),
    },
    "cherenkov_area50_m2": {
        "label": "area",
        "unit": r"m$^2$",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e4, 1e12, 33)),
    },
    "cherenkov_solid_angle50_sr": {
        "label": "solid angle",
        "unit": r"sr",
        "bin": binning_utils.Binning(bin_edges=np.geomspace(1e-6, 1e1, 33)),
    },
}

for dkey in DENSITIES:
    for sk in res:
        fig = sebplt.figure(FIGSIZE)
        ax = sebplt.add_axes(fig=fig, span=(0.15, 0.2, 0.8, 0.75))

        for pk in particle_keys:
            dhist = np.histogram2d(
                x=res[sk][pk]["particle_energy_GeV"],
                y=res[sk][pk][dkey],
                bins=(energy_bin["edges"], DENSITIES[dkey]["bin"]["edges"]),
            )[0]
            ehist = np.histogram(
                res[sk][pk]["particle_energy_GeV"],
                bins=energy_bin["edges"],
            )[0]

            qhist = np.zeros(shape=dhist.shape)
            for ebin in range(energy_bin["num"]):
                if np.sum(ehist[ebin]) > 0:
                    qhist[ebin, :] = dhist[ebin, :] / ehist[ebin]

            ax.pcolormesh(
                energy_bin["edges"],
                DENSITIES[dkey]["bin"]["edges"],
                qhist.T,
                cmap=PCOLORMAPS[pk],
            )

        ax.loglog()
        ax.set_xlabel("energy" + PLT["label_unit_seperator"] + "GeV")
        ax.set_ylabel(
            DENSITIES[dkey]["label"]
            + PLT["label_unit_seperator"]
            + DENSITIES[dkey]["unit"]
        )
        ax.set_ylim(DENSITIES[dkey]["bin"]["limits"])
        ax.set_xlim(energy_bin["limits"])
        fig.savefig(os.path.join(out_dir, "{:s}_{:s}.jpg".format(sk, dkey)))
        sebplt.close(fig)
