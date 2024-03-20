import os
from importlib import resources as importlib_resources
import subprocess


def run_script_job(job):
    return _run_script(script=job["script"], argv=job["argv"])


def run_script(script, argv):
    if not script.endswith(".py"):
        script += ".py"

    script_path = os.path.join(
        importlib_resources.files("magnetic_deflection"), "scripts", script
    )

    args = []
    args.append("python")
    args.append(script_path)
    args += argv
    return subprocess.call(args)
