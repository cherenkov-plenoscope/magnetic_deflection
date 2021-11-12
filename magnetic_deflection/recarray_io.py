import numpy as np
import pandas
import shutil
import io
import tarfile


def write_to_tar(recarray, path):
    with tarfile.open(path + ".tmp", "w") as tarfout:
        for column_key in recarray.dtype.names:
                dtype_key = recarray.dtype[column_key].str
                payload_bytes = recarray[column_key].tobytes()
                tarinfo = tarfile.TarInfo()
                tarinfo.name = "{:s}.{:s}".format(column_key, dtype_key)
                tarinfo.size = len(payload_bytes)
                with io.BytesIO() as fileobj:
                    fileobj.write(payload_bytes)
                    fileobj.seek(0)
                    tarfout.addfile(tarinfo=tarinfo, fileobj=fileobj)
    shutil.move(path + ".tmp", path)


def read_from_tar(path):
    out = {}
    with tarfile.open(path, "r") as tarfin:
        for tarinfo in tarfin:
            column_key, dtype_key = str.split(tarinfo.name, ".")
            level_column_bytes = tarfin.extractfile(tarinfo).read()
            out[column_key] = np.frombuffer(
                level_column_bytes, dtype=dtype_key
            )
    return pandas.DataFrame(out).to_records(index=False)


def write_to_csv(recarray, path):
    df = pandas.DataFrame(recarray)
    csv = df.to_csv(index=False)
    with open(path + ".tmp", "wt") as f:
        f.write(csv)
    shutil.move(path + ".tmp", path)


def read_from_csv(path):
    df = pandas.read_csv(path)
    rec = df.to_records(index=False)
    return rec
