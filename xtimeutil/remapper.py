import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import sparse
import xarray as xr
from scipy.sparse import csr_matrix

from .axis import Axis, _get_time_bounds_dims


# Because we are using the time bounds, using time frequencies anchored at the end 
# would yield incorrect results. Therefore, we should treat all time frequencies to be
# anchored at the beginning all the time. 

_FREQUENCIES = {
    'A': 'AS',
    'AS': 'AS',
    'Y': 'YS',
    'YS': 'YS',
    'Q': 'QS',
    'QS': 'QS',
    'M': 'MS',
    'MS': 'MS',
    'D': 'D',
    'H': 'H',
    'T': 'T',
    'min': 'min',
    'S': 'S',
    'AS-JAN': 'AS-JAN',
    'AS-FEB': 'AS-FEB',
    'AS-MAR': 'AS-MAR',
    'AS-APR': 'AS-APR',
    'AS-MAY': 'AS-MAY',
    'AS-JUN': 'AS-JUN',
    'AS-JUL': 'AS-JUL',
    'AS-AUG': 'AS-AUG',
    'AS-SEP': 'AS-SEP',
    'AS-OCT': 'AS-OCT',
    'AS-NOV': 'AS-NOV',
    'AS-DEC': 'AS-DEC',
    'A-JAN': 'AS-JAN',
    'A-FEB': 'AS-FEB',
    'A-MAR': 'AS-MAR',
    'A-APR': 'AS-APR',
    'A-MAY': 'AS-MAY',
    'A-JUN': 'AS-JUN',
    'A-JUL': 'AS-JUL',
    'A-AUG': 'AS-AUG',
    'A-SEP': 'AS-SEP',
    'A-OCT': 'AS-OCT',
    'A-NOV': 'AS-NOV',
    'A-DEC': 'AS-DEC',
    'QS-JAN': 'QS-JAN',
    'QS-FEB': 'QS-FEB',
    'QS-MAR': 'QS-MAR',
    'QS-APR': 'QS-APR',
    'QS-MAY': 'QS-MAY',
    'QS-JUN': 'QS-JUN',
    'QS-JUL': 'QS-JUL',
    'QS-AUG': 'QS-AUG',
    'QS-SEP': 'QS-SEP',
    'QS-OCT': 'QS-OCT',
    'QS-NOV': 'QS-NOV',
    'QS-DEC': 'QS-DEC',
    'Q-JAN': 'QS-JAN',
    'Q-FEB': 'QS-FEB',
    'Q-MAR': 'QS-MAR',
    'Q-APR': 'QS-APR',
    'Q-MAY': 'QS-MAY',
    'Q-JUN': 'QS-JUN',
    'Q-JUL': 'QS-JUL',
    'Q-AUG': 'QS-AUG',
    'Q-SEP': 'QS-SEP',
    'Q-OCT': 'QS-OCT',
    'Q-NOV': 'QS-NOV',
    'Q-DEC': 'QS-DEC',
}

# Copied from xarray
# https://github.com/pydata/xarray/blob/52ee5df/xarray/coding/cftime_offsets.py
_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = fr'^((?P<multiple>\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$'


def _validate_freq(freq):
    try:
        freq_data = re.match(_PATTERN, freq).groupdict()
    except AttributeError:
        valid_frequencies = sorted(list(set(_FREQUENCIES.keys()) - set(_FREQUENCIES.values())))
        raise ValueError(
            f'Invalid frequency string provided. Valid frequencies (plus their multiples) include: {valid_frequencies}'
        )
    freq = freq_data['freq']
    multiples = freq_data['multiple']
    if multiples is None:
        multiples = 1
    else:
        multiples = int(multiples)
    return f'{multiples}{_FREQUENCIES[freq]}'


class Remapper:
    """ An object that facilitates conversion between two time axis.
    """

    def __init__(self, ds, freq, time_coord_name='time', binding=None):
        """
        Create a new Remapper object that facilitates conversion between two time axis.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant time coordinate information.
        freq : str / frequency object
            The offset object representing target conversion also known as resampling
            frequency (e.g., 'MS', '2D', 'H', or '3T' For full specification of available
            frequencies, please see `here
            <https://xarray.pydata.org/en/stable/generated/xarray.cftime_range.html>`_.
        time_coord_name : str, optional
            Name of the time coordinate, by default 'time'
        binding : {'left', 'right', 'middle'}, optional
            Defines different ways time data tick could be bound to an interval.
            If None (default), attempt at inferring the time data tick binding from the
            input data set.

            - `left`: means that the data tick is bound to the left/beginning of
              the interval or the lower time bound.

            - `right`: means that the data tick is bound to the right/end of the
               interval or the upper time bound.

            - `middle`: means that the data tick is bound half-way through between
               lower time bound and upper time bound.

        """
        self._ds = ds
        self._from_axis = Axis(ds, time_coord_name, binding)
        self.info = self._construct_remapper_info(freq)

    def _construct_remapper_info(self, freq):
        freq = _validate_freq(freq)
        binding = self._from_axis.attrs['binding']
        ti = self._from_axis.decoded_time_bounds.values.flatten().min()
        tf = self._from_axis.decoded_time_bounds.values.flatten().max()

        incoming_time_bounds = self._from_axis.decoded_time_bounds
        outgoing_time_bounds = _generate_outgoing_time_bounds(
            incoming_time_bounds, freq, ti, tf, self._from_axis.attrs
        )

        wgts, col_idx, row_idx = self._get_coverage_matrix(
            incoming_time_bounds, outgoing_time_bounds
        )

        c = xr.Dataset()
        c['weights'] = xr.DataArray(
            sparse.COO.from_scipy_sparse(wgts), dims=['outgoing', 'incoming']
        )
        c['row_idx'] = row_idx
        c['col_idx'] = col_idx
        c['ti'] = ti
        c['tf'] = tf
        c['outgoing_time_bounds'] = outgoing_time_bounds
        c.attrs['freq'] = freq
        c.attrs['binding'] = binding
        c.attrs.update(self._from_axis.attrs)

        return c

    def _get_coverage_matrix(self, incoming_time_bounds, outgoing_time_bounds):
        encoded_time_bounds_in, _, _ = xr.coding.times.encode_cf_datetime(incoming_time_bounds)
        encoded_time_bounds_out, _, _ = xr.coding.times.encode_cf_datetime(outgoing_time_bounds)

        if self._from_axis.attrs['time_bounds_dim_axis_num'] == 1:
            from_lower_bound = encoded_time_bounds_in[:, 0]
            from_upper_bound = encoded_time_bounds_in[:, 1]
            to_lower_bound = encoded_time_bounds_out[:, 0]
            to_upper_bound = encoded_time_bounds_out[:, 1]

        else:
            from_lower_bound = encoded_time_bounds_in[0, :]
            from_upper_bound = encoded_time_bounds_in[1, :]
            to_lower_bound = encoded_time_bounds_out[0, :]
            to_upper_bound = encoded_time_bounds_out[1, :]

        m = to_lower_bound.size
        n = from_lower_bound.size

        row_idx = []
        col_idx = []
        weights = []
        for r in range(m):
            toLB = to_lower_bound[r]
            toUB = to_upper_bound[r]
            toLength = toUB - toLB
            for c in range(n):
                fromLB = from_lower_bound[c]
                fromUB = from_upper_bound[c]
                fromLength = fromUB - fromLB

                if (fromUB <= toLB) or (fromLB >= toUB):  # No coverage
                    continue
                elif (fromLB <= toLB) and (fromUB >= toLB) and (fromUB <= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    fraction_overlap = (fromUB - toLB) / fromLength
                    weights.append(fraction_overlap * (fromLength / toLength))
                elif (fromLB >= toLB) and (fromLB < toUB) and (fromUB >= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    fraction_overlap = (toUB - fromLB) / fromLength
                    weights.append(fraction_overlap * (fromLength / toLength))
                elif (fromLB >= toLB) and (fromUB <= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    fraction_overlap = 1.0
                    weights.append(fraction_overlap * (fromLength / toLength))
                elif (fromLB <= toLB) and (fromUB >= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    fraction_overlap = (toUB - toLB) / fromLength
                    weights.append(fraction_overlap * (fromLength / toLength))

        wgts = csr_matrix((weights, (row_idx, col_idx)), shape=(m, n)).tolil()
        mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
        wgts[mask, 0] = np.nan
        wgts = wgts.tocsr()

        return wgts, col_idx, row_idx

    def _prepare_input_data(self, da):
        n = self.info.weights.data.shape[1]
        data = da.data.copy()
        time_axis = _get_time_axis_dim_num(self._from_axis.attrs, da)
        if data.ndim == 1:
            data = data.reshape((-1, 1))

        if data.shape[time_axis] != n:
            message = f"""The length ({data.shape[time_axis]}) of input time dimension does not
            match to that of the provided remapper ({n})"""
            raise ValueError(message)

        if time_axis != 0:
            data = np.moveaxis(data, time_axis, 0)

        trailing_shape = data.shape[1:]
        data = data.reshape((n, -1))

        return data, trailing_shape

    def _prepare_output_data(self, input_data, output_data, time_axis, trailing_shape):

        shape = (output_data.shape[0], *trailing_shape)
        data = np.moveaxis(output_data.reshape(shape), 0, time_axis)

        original_dims = input_data.dims
        coords = OrderedDict()
        dims = []
        for dim in original_dims:
            if dim != self._from_axis.attrs['time_coord_name']:
                if dim in input_data.coords:
                    coords[dim] = input_data.coords[dim]
                    dims.append(dim)
            else:

                times = self._from_axis._bindings[self.info.attrs['binding']](
                    self.info.outgoing_time_bounds,
                    axis=self.info.attrs['time_bounds_dim_axis_num'],
                )
                times = xr.DataArray(times)
                coords[dim] = xr.DataArray(
                    times,
                    coords={self.info.attrs['time_coord_name']: times},
                    attrs=input_data.attrs,
                )
                dims.append(dim)

        return xr.DataArray(data, dims=dims, coords=coords)

    def average(self, da):
        time_axis = _get_time_axis_dim_num(self._from_axis.attrs, da)
        input_data, trailing_shape = self._prepare_input_data(da)
        nan_mask = np.isnan(input_data)
        non_nan_mask = np.ones(input_data.shape, dtype=np.int8)
        non_nan_mask[nan_mask] = 0
        input_data[nan_mask] = 0
        output_data = np.dot(self.info.weights.data, input_data)
        output_data = self._prepare_output_data(da, output_data, time_axis, trailing_shape)
        return output_data

    mean = average 



def _generate_outgoing_time_bounds(incoming_time_bounds, freq, ti, tf, attrs):

    warning_message = f'Resample frequency is greater than extent of incoming time axis. Doubling time axis interval.'

    if xr.core.common.is_np_datetime_like(incoming_time_bounds.dtype):
        # Use to_offset() function to compute offset that allows us to generate
        # time range that includes the end of the incoming time bounds.
        offset = pd.tseries.frequencies.to_offset(freq)

        time_bounds = pd.date_range(start=pd.to_datetime(ti), end=pd.to_datetime(tf), freq=freq)

        if (len(time_bounds) == 1) or (time_bounds[-1] < tf):
            # this should be rare
            if len(time_bounds) == 1:
                warnings.warn(warning_message)

            time_bounds = pd.date_range(
                start=pd.to_datetime(ti), end=pd.to_datetime(tf) + offset, freq=freq,
            )

    else:
        offset = xr.coding.cftime_offsets.to_offset(freq)
        time_bounds = xr.cftime_range(start=ti, end=tf, freq=freq, calendar=attrs['calendar'],)

        if (len(time_bounds) == 1) or (time_bounds[-1] < tf):
            # this should be rare
            if len(time_bounds) == 1:
                warnings.warn(warning_message)

            time_bounds = xr.cftime_range(
                start=ti, end=tf + offset, freq=freq, calendar=attrs['calendar'],
            )

    msg = f"""{tf} upper bound from the incoming time axis is not covered in the outgoing
    time axis which has {time_bounds[-1]} as the upper bound."""

    assert time_bounds[-1] >= tf, msg
    outgoing_time_bounds = np.vstack((time_bounds[:-1], time_bounds[1:])).T
    dims = _get_time_bounds_dims(attrs)

    if attrs['time_bounds_dim_axis_num'] == 0:
        outgoing_time_bounds = outgoing_time_bounds.T

    return xr.DataArray(dims=dims, data=outgoing_time_bounds)


def _get_time_axis_dim_num(attrs, da):
    """
    Return the dimension number of the time axis coordinate in a DataArray.
    """
    time_coord_name = attrs['time_coord_name']
    return da.get_axis_num(time_coord_name)
