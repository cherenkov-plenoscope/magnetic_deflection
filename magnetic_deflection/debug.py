import os
import glob

from . import tools
from . import work_dir_structure


def list_incomplete_jobs(work_dir):
    jobs_path_state = list_incomplete_job_paths_and_states(work_dir)
    jobs = []
    for job_path_state in jobs_path_state:
        job_path, job_state = job_path_state
        print("job", job_path, "state:", job_state)
        job = tools.read_json(job_path)
        jobs.append(job)
    return jobs


def list_incomplete_job_paths_and_states(work_dir):
    CFG = work_dir_structure.read_config(work_dir, ["sites", "particles"], )

    incomplete = []
    for skey in CFG["sites"]:
        for pkey in CFG["particles"]:
            sp_dir = os.path.join(work_dir, "map", skey, pkey)

            job_ids = glob_job_ids(os.path.join(sp_dir, "*_job.json"))
            print("Search ", len(job_ids), " jobs in ", sp_dir)

            for job_id in job_ids:
                job_path = os.path.join(
                    sp_dir, "{:06d}_job.json".format(job_id)
                )
                log_path = os.path.join(
                    sp_dir, "{:06d}_log.jsonl".format(job_id)
                )

                if os.path.exists(log_path):
                    job_log = tools.read_jsonl(log_path)
                    if not job_log_is_complete(job_log=job_log):
                        incomplete.append((job_path, "log incomplete"))
                else:
                    incomplete.append((job_path, "no log"))

    return incomplete


def job_log_is_complete(job_log):
    if len(job_log) == 0:
        return False

    finale_log = job_log[-1]
    if "m" not in finale_log:
        return False

    finale_message = finale_log["m"]
    if finale_message != "job: end":
        return False

    return True


def glob_job_ids(wildcard_path):
    paths = glob.glob(wildcard_path)
    basenames = [os.path.basename(p) for p in paths]
    ids = [int(p[0:6]) for p in basenames]
    return ids
