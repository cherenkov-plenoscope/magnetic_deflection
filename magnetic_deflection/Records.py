import pandas
import array

MAP_NUMPY_DTYPES_ON_ARRAY_CTYPES = {
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
        ctype = MAP_NUMPY_DTYPES_ON_ARRAY_CTYPES[dtype_key]
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
    return pandas.DataFrame(records).to_records(index=False)
