import rename_after_writing as rnw
import os
import glob
import uuid
import json_line_logger


def init(production_dir):
    os.makedirs(production_dir, exist_ok=True)
    with rnw.open(os.path.join(production_dir, "lock.open"), "wt") as f:
        pass

    with rnw.open(os.path.join(production_dir, "next_run_id.txt"), "wt") as f:
        f.write("1")

    os.makedirs(os.path.join(production_dir, "logs"), exist_ok=True)


class Production:
    def __init__(self, production_dir):
        self.production_dir = production_dir
        self.uuid = str(uuid.uuid4())
        self.log = json_line_logger.LoggerFile(
            path=os.path.join(
                self.production_dir, "logs", "{:s}.jsonl".format(self.uuid)
            ),
            name=self.uuid,
        )

    def _read_lock_uuid(self):
        lock_paths = glob.glob(os.path.join(self.production_dir, "lock.*"))
        assert len(lock_paths) == 1
        lock_basename = os.path.basename(lock_paths[0])
        lock_uuid = str.split(lock_basename, ".")[1]
        return lock_uuid

    def lock_is_mine(self):
        lock_uuid = self._read_lock_uuid()
        if lock_uuid == self.uuid:
            return True
        else:
            return False

    def lock_is_open(self):
        lock_uuid = self._read_lock_uuid()
        if lock_uuid == "open":
            return True
        else:
            return False

    def lock(self):
        self.log.info("Try to get lock.")
        if not self.lock_is_open():
            msg = "Can not get lock becasue lock is not open."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can get lock because it is open.")

        os.rename(
            os.path.join(self.production_dir, "lock.open"),
            os.path.join(self.production_dir, "lock.{:s}".format(self.uuid)),
        )
        self.log.info("Lock is mine.")

    def unlock(self):
        self.log.info("Try to open lock.")
        if not self.lock_is_mine():
            msg = "Can not open lock becasue lock is not mine."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can open lock because it is mine.")

        os.rename(
            os.path.join(self.production_dir, "lock.{:s}".format(self.uuid)),
            os.path.join(self.production_dir, "lock.open"),
        )
        self.log.info("Lock is open.")

    def get_next_run_id_and_bumb(self):
        self.log.info("Try to bumb run_id.")
        if not self.lock_is_mine():
            msg = "The lock is not mine. I can not bumb the run_id."
            self.log.error(msg)
            raise AssertionError(msg)
        self.log.info("Can bumb run_id because the lock is mine")

        path = os.path.join(self.production_dir, "next_run_id.txt")
        with open(path, "rt") as f:
            next_run_id = int(f.read())

        self.log.info("Next run_id is {:d}.".format(next_run_id))

        next_next_run_id = next_run_id + 1
        with open(path, "wt") as f:
            f.write("{:d}".format(next_next_run_id))

        self.log.info(
            "Bumbed the next next run_id to {:d}.".format(next_next_run_id)
        )

        return next_run_id


def default_corsika_primary_mod_path():
    return os.path.join(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd",
    )
