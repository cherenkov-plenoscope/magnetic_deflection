from magnetic_deflection import dynamicsizerecarray
import pytest
import numpy as np


def test_init_no_parameters():
    with pytest.raises(AttributeError):
        dra = dynamicsizerecarray.DynamicSizeRecarray()


def test_init_conflicting_parameters():
    with pytest.raises(AttributeError):
        dra = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=[("key", "i8")],
            recarray=np.core.records.recarray(
                shape=0,
                dtype=[("key", "i8")],
            ),
        )


def test_init_from_dtype():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )


def test_init_from_recarray():
    for size in [0, 2, 100]:
        recarray = np.core.records.recarray(
            shape=size,
            dtype=[("a", "i8"), ("b", "u2")],
        )

        dra = dynamicsizerecarray.DynamicSizeRecarray(recarray=recarray)

        assert dra.size == size


def test_append_record():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )

    SIZE = 1024
    for i in range(SIZE):
        record = {"a": 2 * i, "b": i}
        dra.append_record(record=record)
        assert len(dra) == i + 1
        assert dra.size == len(dra)

    out = dra.to_recarray()
    assert len(out) == SIZE
    assert out["a"].dtype == np.int64
    assert out["b"].dtype == np.uint16

    for j in range(SIZE):
        assert out["a"][j] == 2 * j
        assert out["b"][j] == j


def test_append_recarray():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )

    NUM_BLOCKS = 12
    BLOCK_SIZE = 8
    for i in range(NUM_BLOCKS):
        recarray = np.core.records.recarray(
            shape=BLOCK_SIZE,
            dtype=[("a", "i8"), ("b", "u2")],
        )
        recarray["a"] = 2 * i * np.ones(BLOCK_SIZE)
        recarray["b"] = i * np.ones(BLOCK_SIZE)

        dra.append_recarray(recarray=recarray)

        assert len(dra) == (1 + i) * BLOCK_SIZE
        assert dra.size == len(dra)

    out = dra.to_recarray()
    assert len(out) == NUM_BLOCKS * BLOCK_SIZE
    assert out["a"].dtype == np.int64
    assert out["b"].dtype == np.uint16

    for i in range(NUM_BLOCKS):
        for j in range(BLOCK_SIZE):
            k = i * BLOCK_SIZE + j
            assert out["a"][k] == 2 * i
            assert out["b"][k] == i


def test_to_recarray_when_empty():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )

    out = dra.to_recarray()

    assert len(out) == 0
    assert out["a"].dtype == np.int64
    assert out["b"].dtype == np.uint16


def test_str_repr():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )
    s = str(dra)
    assert len(s) > 0
