import re
import warnings

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

from .axis import BINDINGS, get_time_axis_info

_OUTGOING_KEY = 'outgoing'
_INCOMING_KEY = 'incoming'

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
        freq = _validate_freq(freq)
        self.incoming = get_time_axis_info(ds, time_coord_name, binding)
        ti = self.incoming['decoded_time_bounds'].values.flatten().min()
        tf = self.incoming['decoded_time_bounds'].values.flatten().max()
        self.info = {'ti': ti, 'tf': tf, 'freq': freq, 'binding': self.incoming['binding']}
        self.outgoing = self.get_outgoing_time_axis_info()
        self.coverage = _get_coverage_info(
            self.incoming['encoded_time_bounds'].values,
            self.outgoing['encoded_time_bounds'].values,
            self.incoming['time_bounds_dim_axis_num'],
        )

        _coords = {
            _OUTGOING_KEY: self.outgoing['decoded_times'].data,
            _INCOMING_KEY: self.incoming['decoded_times'].data,
        }
        self.weights = construct_coverage_matrix(
            self.coverage['weights'],
            self.coverage['col_idx'],
            self.coverage['row_idx'],
            self.coverage['shape'],
            coords=_coords,
        )

    def get_outgoing_time_axis_info(self):
        attrs = {'units': self.incoming['units'], 'calendar': self.incoming['calendar']}
        attrs.update(self._ds[self.incoming['time_coord_name']].attrs)
        outgoing_decoded_time_bounds = _construct_outgoing_time_bounds(
            self.info['freq'],
            self.info['ti'],
            self.info['tf'],
            attrs['calendar'],
            self.incoming['np_datetime_like'],
            self.incoming['time_bounds_dim_axis_num'],
            self.incoming['time_bounds_dims'],
        )
        time_bounds, _, _ = xr.coding.times.encode_cf_datetime(
            outgoing_decoded_time_bounds, units=attrs['units'], calendar=attrs['calendar']
        )
        times = BINDINGS[self.info['binding']](
            time_bounds, axis=self.incoming['time_bounds_dim_axis_num']
        )
        outgoing_encoded_times = xr.DataArray(
            times,
            attrs=attrs,
            dims=[self.incoming['time_coord_name']],
            coords={self.incoming['time_coord_name']: times},
        )
        outgoing_encoded_time_bounds = xr.DataArray(
            time_bounds,
            dims=outgoing_decoded_time_bounds.dims,
            coords={self.incoming['time_coord_name']: times},
        )
        decoded_times = xr.coding.times.decode_cf_datetime(
            times,
            units=attrs['units'],
            calendar=attrs['calendar'],
            use_cftime=self.incoming['use_cftime'],
        )
        outgoing_decoded_times = xr.DataArray(
            decoded_times,
            attrs=attrs,
            dims=[self.incoming['time_coord_name']],
            coords={self.incoming['time_coord_name']: decoded_times},
        )
        x = {
            'decoded_time_bounds': outgoing_decoded_time_bounds,
            'encoded_time_bounds': outgoing_encoded_time_bounds,
            'encoded_times': outgoing_encoded_times,
            'decoded_times': outgoing_decoded_times,
        }
        x.update(attrs)
        return x

    def mean(self, data):
        indata = data.copy()
        _sanitize_input_data(indata, self.incoming['time_coord_name'], self.weights)
        if isinstance(indata, xr.DataArray):
            if isinstance(indata.data, dask.array.Array):
                incoming_time_chunks = dict(zip(indata.dims, indata.chunks))[
                    self.incoming['time_coord_name']
                ][0]
                return _mean(
                    self.weights.chunk({_OUTGOING_KEY: incoming_time_chunks}),
                    indata,
                    self.incoming['time_coord_name'],
                )
            else:
                return _mean(self.weights, indata, self.incoming['time_coord_name'])

    average = mean


def _mean(weights, indata, time_coord_name):
    indata = indata.rename({time_coord_name: _INCOMING_KEY})
    indata = indata.where(indata.notnull(), 0)
    out = xr.dot(weights, indata).rename({_OUTGOING_KEY: time_coord_name})
    return out.where(out != 0)


def construct_coverage_matrix(weights, col_idx, row_idx, shape, coords={}):
    wgts = csr_matrix((weights, (row_idx, col_idx)), shape=shape).tolil()
    mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
    wgts[mask, 0] = np.nan
    weights = xr.DataArray(wgts.toarray(), dims=[_OUTGOING_KEY, _INCOMING_KEY], coords=coords)
    return weights


def _get_coverage_info(
    incoming_encoded_time_bounds, outgoing_encoded_time_bounds, time_bounds_dim_axis_num
):

    if time_bounds_dim_axis_num == 1:
        incoming_lower_bounds = incoming_encoded_time_bounds[:, 0]
        incoming_upper_bounds = incoming_encoded_time_bounds[:, 1]
        outgoing_lower_bounds = outgoing_encoded_time_bounds[:, 0]
        outgoing_upper_bounds = outgoing_encoded_time_bounds[:, 1]
    else:
        incoming_lower_bounds = incoming_encoded_time_bounds[0, :]
        incoming_upper_bounds = incoming_encoded_time_bounds[1, :]
        outgoing_lower_bounds = outgoing_encoded_time_bounds[0, :]
        outgoing_upper_bounds = outgoing_encoded_time_bounds[1, :]

    n = incoming_lower_bounds.size
    m = outgoing_lower_bounds.size

    row_idx = []
    col_idx = []
    weights = []
    for r in range(m):
        toLB = outgoing_lower_bounds[r]
        toUB = outgoing_upper_bounds[r]
        toLength = toUB - toLB
        for c in range(n):
            fromLB = incoming_lower_bounds[c]
            fromUB = incoming_upper_bounds[c]
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

    coverage = {'weights': weights, 'col_idx': col_idx, 'row_idx': row_idx, 'shape': (m, n)}
    return coverage


def _construct_outgoing_time_bounds(
    freq, ti, tf, calendar, np_datetime_like, time_bounds_dim_axis_num, time_bounds_dims, attrs={}
):

    warning_message = f'Resample frequency={freq} is greater than extent of incoming time axis. Doubling outgoing time axis interval.'

    if np_datetime_like:
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
        time_bounds = xr.cftime_range(start=ti, end=tf, freq=freq, calendar=calendar,)
        if (len(time_bounds) == 1) or (time_bounds[-1] < tf):
            # this should be rare
            if len(time_bounds) == 1:
                warnings.warn(warning_message)

            time_bounds = xr.cftime_range(start=ti, end=tf + offset, freq=freq, calendar=calendar,)

    msg = (
        f'{tf} upper bound from the incoming time axis is not covered in the outgoing '
        f'time axis which has {time_bounds[-1]} as the upper bound.'
    )

    assert time_bounds[-1] >= tf, msg
    outgoing_time_bounds = np.vstack((time_bounds[:-1], time_bounds[1:])).T

    if time_bounds_dim_axis_num == 0:
        outgoing_time_bounds = outgoing_time_bounds.T

    out = xr.DataArray(dims=time_bounds_dims, data=outgoing_time_bounds)
    out.attrs = attrs
    return out


def _sanitize_input_data(data, time_coord_name, weights):
    message = (
        f'The length ({data[time_coord_name].size}) of incoming time dimension does not match '
        f"with the provided remapper object's incoming time dimension ({weights[_INCOMING_KEY].size})"
    )
    assert data[time_coord_name].size == weights[_INCOMING_KEY].size, message
