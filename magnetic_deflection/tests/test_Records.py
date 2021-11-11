from magnetic_deflection import Records
import numpy
import pandas

def _list_of_dicts_to_records(lod):
    return pandas.DataFrame(lod).to_records(index=False)


def close(a, b, epsilon=1e-6):
    return numpy.abs(a - b) < epsilon


def test_init():
    A = Records.init(dtypes={"a": "i4", "b": "f4"})
    assert "a" in A
    assert "b" in A
    assert len(A.keys()) == 2

    assert len(A["a"]) == 0
    assert len(A["b"]) == 0


def test_append():
    A = Records.init(dtypes={"a": "i4", "b": "f4"})
    A = Records.append_dict(A, {"a": 4, "b": 1.337})
    assert len(A["a"]) == 1
    assert len(A["b"]) == 1


def test_to_numpy_recarray():
    A = Records.init(dtypes={"a": "i4", "b": "f4"})
    A = Records.append_dict(A, {"a": 4, "b": 2.5})
    R = Records.to_numpy_recarray(A)
    assert R["a"][0] == 4
    assert R.dtype["a"] == numpy.int32

    assert R["b"][0] == 2.5
    assert R.dtype["b"] == numpy.float32


def test_append_numpy_recarray():
    A = Records.init(dtypes={"a": "i4", "b": "f4"})
    A = Records.append_dict(A, {"a": 4, "b": 2.5})

    R = _list_of_dicts_to_records(
        [
            {"a": 2, "b": 1.3},
            {"a": 42, "b": 4.2},
        ],
    )

    A = Records.append_numpy_recarray(A, R)

    assert len(A["a"]) == 3
    assert len(A["b"]) == 3

    assert A["a"][0] == 4
    assert A["a"][1] == 2
    assert A["a"][2] == 42

    assert close(A["b"][0], 2.5)
    assert close(A["b"][1], 1.3)
    assert close(A["b"][2], 4.2)
