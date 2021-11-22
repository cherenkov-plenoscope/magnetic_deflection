import os
from . import tools


STRUCTURE = {
    "config/sites.json": {},
    "config/particles.json": {},
    "config/pointing.json": {},
    "config/config.json": {},
    "config/plotting.json": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_job.json": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_log.jsonl": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_discovery.jsonl": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_deflection.json": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_statistics.recarray.tar": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_statistics_steering.tar": {},
    "reduce/{site_key:s}/{particle_key:s}/deflection.csv": {},
    "reduce/{site_key:s}/{particle_key:s}/statistics.recarray.tar": {},
    "reduce/{site_key:s}/{particle_key:s}/statistics_steering.tar": {},
}


def all_config_keys():
    out = []
    for path in STRUCTURE:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        key, ext = os.path.splitext(basename)
        if dirname == "config" and ext == ".json":
            out.append(key)
    return out


def read_config(work_dir, keys=all_config_keys()):
    CFG = {}
    for key in keys:
        CFG[key] = tools.read_json(
            os.path.join(work_dir, "config", key + ".json")
        )
    return CFG


def map_basenames_format():
    out = {}
    for path in STRUCTURE:
        if "map/{site_key:s}/{particle_key:s}/{job_id:06d}" in path:
            basename = os.path.basename(path)
            name = str.replace(basename, "{job_id:06d}_", "")
            key = str.split(name, ".")[0]
            out[key] = basename
    return out


def map_basenames_wildcard():
    """
    This is not a regular expression. This is shell style glob.
    """
    formats = map_basenames_format()
    out = {}
    for key in formats:
        b = formats[key]
        out[key] = str.replace(b, "{job_id:06d}", "[0-9]" * 6)
    return out


def map_basenames(job_id):
    assert job_id >= 0
    formats = map_basenames_format()
    out = {}
    for key in formats:
        out[key] = formats[key].format(job_id=job_id)
    return out


def map_paths(map_dir, job_id):
    basenames = map_basenames(job_id)
    out = {}
    for key in basenames:
        out[key] = os.path.join(map_dir, basenames[key])
    return out


def reduce_basenames():
    out = {}
    for path in STRUCTURE:
        if "reduce/{site_key:s}/{particle_key:s}" in path:
            basename = os.path.basename(path)
            key = str.split(basename, ".")[0]
            out[key] = basename
    return out
