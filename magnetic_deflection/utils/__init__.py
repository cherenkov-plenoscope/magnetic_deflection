from . import scripts

import atmospheric_cherenkov_response
import os
import glob
import numpy as np
import tarfile
import rename_after_writing as rnw


def list_site_keys_in_path(path):
    potential_keys = _list_dirnames_in_path(path=path)
    known_keys = atmospheric_cherenkov_response.sites.keys()
    return _filter_keys(
        keys=potential_keys,
        keys_to_keep=known_keys,
    )


def list_particle_keys_in_path(path):
    potential_keys = _list_dirnames_in_path(path=path)
    known_keys = atmospheric_cherenkov_response.particles.keys()
    return _filter_keys(
        keys=potential_keys,
        keys_to_keep=known_keys,
    )


def _get_common_sites_and_particles(tree):
    site_keys = list(tree.keys())
    particle_keys = set.intersection(*[set(tree[sk]) for sk in tree])
    particle_keys = list(particle_keys)
    # sort for reproducibility
    site_keys = sorted(site_keys)
    particle_keys = sorted(particle_keys)
    return site_keys, particle_keys


def _sniff_site_and_particle_keys(work_dir):
    site_keys = list_site_keys_in_path(path=work_dir)
    tree = {}
    for sk in site_keys:
        tree[sk] = list_particle_keys_in_path(path=os.path.join(work_dir, sk))
    return tree


def _list_dirnames_in_path(path):
    dirs = glob.glob(os.path.join(path, "*"))
    dirnames = []
    for dd in dirs:
        if os.path.isdir(dd):
            dirnames.append(os.path.basename(dd))
    return dirnames


def _filter_keys(keys, keys_to_keep):
    out = []
    for key in keys:
        if key in keys_to_keep:
            out.append(key)
    return out


def gauss1d(x, mean, sigma):
    return np.exp((-1 / 2) * ((x - mean) ** 2) / (sigma**2))


def estimate_num_bins_to_contain_quantile(
    counts, q, mode="min_num_bins", bin_apertures=None
):
    """
    Parameters
    ----------
    counts : array_like, ints / uints >= 0
        The counting content of bins. Counts must be >= 0
    q : float
        Quantile
    mode : str
        To either return tne 'min_num_bins' or the 'max_num_bins' which
        is required to contain the desired quantile.
    bin_apertures : array_like or None
        The apertures of the bins. Default is ones.

    Returns
    -------
    num_bins : float
        The number of bins which must be taken into account to contain the
        desired quantile.
    (num_bins, aperture) : (float, float)
        If the parameter bin_apertures is not None, the accumulated apertue
        of the bins (related to num_bins) is returned, too.
    """
    assert 0.0 <= q <= 1.0
    counts = np.asarray(counts, dtype=int)
    assert np.all(counts >= 0)

    if bin_apertures is None:
        APERTURE = False
        bin_apertures = np.ones(len(counts))
    else:
        APERTURE = True
        bin_apertures = np.asarray(bin_apertures)
        assert len(bin_apertures) == len(counts)

    if mode == "min_num_bins":
        mode_factor = -1
    elif mode == "max_num_bins":
        mode_factor = +1
    else:
        raise ValueError("Unknown mode '{:s}'.".format(mode))

    ars = np.argsort(mode_factor * counts)

    sorted_counts = counts[ars]
    sorted_bin_apertures = bin_apertures[ars]

    total = np.sum(sorted_counts)
    if total == 0:
        if APERTURE:
            return float("nan"), float("nan")
        else:
            return float("nan")

    target = total * q
    part = 0
    numbins = 0
    aperture = 0.0
    for i in range(len(counts)):
        if part + sorted_counts[numbins] < target:
            part += sorted_counts[numbins]
            aperture += sorted_bin_apertures[numbins]
            numbins += 1
        else:
            break
    missing = target - part
    assert missing <= sorted_counts[numbins]
    frac = missing / sorted_counts[numbins]

    aperture += frac * sorted_bin_apertures[numbins]
    numbins += frac

    if APERTURE:
        return numbins, aperture
    else:
        return numbins


def write_array(path, a):
    with rnw.open(path, "wb") as f:
        dtypeline = a.dtype.str + "\n"
        f.write(str.encode(dtypeline))

        shapeline = str.join(",", [str(i) for i in a.shape])
        shapeline += "\n"
        f.write(str.encode(shapeline))

        f.write(a.tobytes(order="c"))


def read_array(path):
    with open(path, "rb") as f:
        dtypeline = f.readline()
        dtypeline = str(dtypeline, encoding="ascii")
        dtypeline = str.strip(dtypeline, "\n")
        dtype = dtypeline

        shapeline = f.readline()
        shapeline = str(shapeline, encoding="ascii")
        shapeline = str.strip(shapeline, "\n")
        shape = [int(s) for s in str.split(shapeline, ",")]

        a = np.frombuffer(buffer=f.read(), dtype=dtype)
    return a.reshape(shape)
