STRUCTURE = {
    "config": {},
    "config/sites.json": {},
    "config/particles.json": {},
    "config/pointing.json": {},
    "config/plotting.json": {},
    "config/config.json": {},
    "map": {},
    "map/{site_key:s}": {},
    "map/{site_key:s}/{particle_key:s}": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_job.json": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_log.jsonl": {},
    "map/{site_key:s}/{particle_key:s}/{job_id:06d}_result.json": {},
    "config/results": {},
    "config/results/{site_key:s}_{particle_key:s}.csv": {},
}

def join(work_dir, *args):
    relpath = os.path.join(*args)
    if relpath not in STRUCTURE:
        raise Warning("The path '{:s}' is not part of the work-dir.".format(relpath))
    return os.path.join(d, *args)
