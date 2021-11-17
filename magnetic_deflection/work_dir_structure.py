import os


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
    "reduce/{site_key:s}/{particle_key:s}/deflection.csv": {},
    "reduce/{site_key:s}/{particle_key:s}/statistics.recarray.tar": {},
}


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
