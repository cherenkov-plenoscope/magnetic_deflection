import magnetic_deflection as mdfl
import numpy as np
import tempfile
import os
import rename_after_writing as rnw


def test_io():
    DTYPE = [("a", "i8"), ("b", "u2")]

    with tempfile.TemporaryDirectory(prefix="magnetic_deflection_") as tmp_dir:
        a = np.core.records.recarray(
            shape=18,
            dtype=DTYPE,
        )

        a_path = os.path.join(tmp_dir, "a.rec")
        with rnw.open(a_path, "wb") as f:
            f.write(a.tobytes())

        with open(a_path, "rb") as f:
            b = np.fromstring(f.read(), dtype=DTYPE)

        for key in a.dtype.names:
            assert key in b.dtype.names
            assert a[key].dtype == b[key].dtype

        np.testing.assert_array_equal(a, b)
