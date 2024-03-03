import os
import numpy as np
import rename_after_writing as rnw


def init(dtype, size):
    return np.core.records.recarray(
        shape=size,
        dtype=dtype,
    )


def write(path, x):
    with rnw.open(path, "wb") as f:
        f.write(x.tobytes())


def read(path, dtype):
    with open(path, "rb") as f:
        x = np.fromstring(f.read(), dtype=dtype)
    return x


def num_records_in_path(path, dtype):
    stat = os.stat(path)
    size_in_bytes = stat.st_size
    return size_in_bytes // size_of_record_in_bytes(dtype=dtype)


def size_of_record_in_bytes(dtype):
    rr = np.core.records.recarray(
        shape=1,
        dtype=dtype,
    )
    return len(rr.tobytes())
