import atmospheric_cherenkov_response
import os
import glob


def _get_common_sites_and_particles(tree):
    site_keys = list(tree.keys())
    particle_keys = set.intersection(*[set(tree[sk]) for sk in tree])
    particle_keys = list(particle_keys)
    # sort for reproducibility
    site_keys = sorted(site_keys)
    particle_keys = sorted(particle_keys)
    return site_keys, particle_keys


def _sniff_site_and_particle_keys(work_dir):
    potential_site_keys = _list_dirnames_in_path(path=work_dir)
    known_site_keys = atmospheric_cherenkov_response.sites.keys()
    known_particle_keys = atmospheric_cherenkov_response.particles.keys()

    site_keys = _filter_keys(
        keys=potential_site_keys,
        keys_to_keep=known_site_keys,
    )

    tree = {}
    for sk in site_keys:
        _potential_particle_keys = _list_dirnames_in_path(
            path=os.path.join(work_dir, sk)
        )
        _particle_keys = _filter_keys(
            keys=_potential_particle_keys,
            keys_to_keep=known_particle_keys,
        )
        tree[sk] = _particle_keys

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
