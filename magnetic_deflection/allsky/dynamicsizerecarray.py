import numpy as np
import copy


class DynamicSizeRecarray:
    def __init__(self, recarray=None, dtype=None):
        """
        A dynamic version of numpy.core.records.recarray.
        Either provide an existing recarray 'recarray' or
        provide the 'dtype' to start with an empty recarray.

        Parameters
        ----------
        recarray : numpy.core.records.recarray, default=None
            The start of the dynamic recarray.
        dtype : list(tuple("key", "dtype_str")), default=None
            The dtype of the dynamic recarray.
        """
        self.size = 0
        if recarray is None and dtype == None:
            raise AttributeError("Requires either 'recarray' or 'dtype'.")
        if recarray is not None and dtype is not None:
            raise AttributeError(
                "Expected either one of 'recarray' or' dtype' to be 'None'"
            )

        if dtype:
            recarray = np.core.records.recarray(
                shape=0,
                dtype=dtype,
            )

        initial_capacity = np.max([2, len(recarray)])
        self.recarray = np.core.records.recarray(
            shape=initial_capacity,
            dtype=recarray.dtype,
        )
        self.append_recarray(recarray=recarray)

    def capacity(self):
        """
        Returns the capacity (in number of records) of the allocated memeory.
        This is the length of the internal recarray.
        """
        return len(self.recarray)

    def to_recarray(self):
        """
        Exports to a numpy.core.records.recarray.
        """
        out = np.core.records.recarray(
            shape=self.size,
            dtype=self.recarray.dtype,
        )
        out = self.recarray[0:self.size]
        return out

    def append_record(self, record):
        """
        Append one record to the dynamic racarray.
        The size of the dynamic recarray will increase by one.

        Parameters
        ----------
        record : dict
            The values from the record-dict for the keys in the recarray will
            be appended to the recarray.
        """
        self._grow_if_needed(additional_size=1)
        for key in self.recarray.dtype.names:
            self.recarray[self.size][key] = record[key]
        self.size += 1

    def append_recarray(self, recarray):
        """
        Append a recarray to the dynamic racarray.
        The size of the dynamic recarray will increase by len(recarray).

        Parameters
        ----------
        recarray : numpy.core.records.recarray
            This will be appended to the internal, dynamic recarray.
        """
        self._grow_if_needed(additional_size=len(recarray))
        start = self.size
        stop = start + len(recarray)
        self.recarray[start:stop] = recarray
        self.size += len(recarray)

    def _grow_if_needed(self, additional_size):
        assert additional_size >= 0
        current_capacity = self.capacity()
        required_size = self.size + additional_size

        if required_size > current_capacity:
            swp = copy.deepcopy(self.recarray)
            next_capacity = np.max([current_capacity * 2, required_size])
            self.recarray = np.core.records.recarray(
                shape=next_capacity,
                dtype=swp.dtype,
            )
            start = 0
            stop = self.size
            self.recarray[start:stop] = swp[0:self.size]
            del swp

    def __len__(self):
        return self.size

    def __repr__(self):
        out = "{:s}(size={:d}, capacity={:d})".format(
            self.__class__.__name__,
            self.size,
            self.capacity(),
        )
        return out
