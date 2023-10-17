import numpy as np
import array

DTYPES2CTYPES = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "i8": "q",
    "u8": "Q",
    "f4": "f",
    "f8": "d",
}


def init(dtypes={"a": "i8"}):
    """
    Returns dict of arrays with key and dtype according to 'dtypes'.
    A Record is a list of dict-objects which can grow fast and space-efficient
    by constraining the dtypes of the fields in the dicts.
    Every dict-object must have the same keys.

    parameter
    ---------

    dtypes : dict
        Maps names, i.e. keys in out-dict, to numpy dtype strings.
    """
    records = {}
    for key in dtypes:
        dtype_key = str(dtypes[key])
        dtype_key = str.replace(dtype_key, "<", "")
        dtype_key = str.replace(dtype_key, ">", "")
        dtype_key = str.replace(dtype_key, "|", "")
        ctype = DTYPES2CTYPES[dtype_key]
        records[key] = array.array(ctype, [])
    return records


def append_dict(records, dict_object):
    for key in records:
        try:
            records[key].append(dict_object[key])
        except Exception as err:
            print(key)
            raise err
    return records


def append_numpy_recarray(records, recarray):
    for key in records:
        try:
            records[key].extend(recarray[key])
        except Exception as err:
            print(key)
            raise err
    return records


def to_numpy_recarray(records):
    CTYPE2DTYPE = _make_ctype2dtype()

    # make dtype for recarray
    recarray_dtype = []
    for key in records:
        key_ctype = records[key].typecode
        key_dtype = CTYPE2DTYPE[key_ctype]
        recarray_dtype.append((key, key_dtype))

    lens = []
    for key in records:
        lens.append(len(records[key]))
    lens = np.array(lens)
    if len(lens) > 0:
        assert np.all(lens[0] == lens)
        size = lens[0]
    else:
        size = 0

    out = np.core.records.recarray(
        shape=size,
        dtype=recarray_dtype,
    )

    # copy records into recarray
    for key in records:
        out[key] = np.array(records[key])

    return out


def get_dtypes_from_numpy_recarray(recarray):
    out = {}
    for key in recarray.dtype.names:
        out[key] = recarray.dtype[key].str
    return out


def _make_ctype2dtype():
    ctype2dtype = {}
    for dtype in DTYPES2CTYPES:
        ctype = DTYPES2CTYPES[dtype]
        ctype2dtype[ctype] = dtype
    return ctype2dtype
