import os
import numpy as np
import rename_after_writing as rnw


class RecarrayUtils:
    def __init__(self, dtype):
        self.dtype = dtype

    def init(self, size):
        return np.core.records.recarray(
            shape=size,
            dtype=self.dtype,
        )

    def write(self, path, x):
        assert x.dtype == self.dtype
        with rnw.open(path, "wb") as f:
            f.write(x.tobytes())

    def read(self, path):
        with open(path, "rb") as f:
            x = np.fromstring(f.read(), dtype=self.dtype)
        return x

    def num_records_in_path(self, path):
        stat = os.stat(path)
        size_in_bytes = stat.st_size
        return size_in_bytes // self.size_of_record_in_bytes()

    def size_of_record_in_bytes(self):
        rr = np.core.records.recarray(
            shape=1,
            dtype=self.dtype,
        )
        return len(rr.tobytes())
