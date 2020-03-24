import numpy as np
import xarray as xr


class Axis:
    """
    An object that represents a time coordinate axis.
    """

    _bindings = {'left': np.min, 'right': np.max, 'middle': np.mean}

    def __init__(self, ds, time_coord_name='time', binding=None):
        """
        Create a new Axis object from an input dataset

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant time coordinate information.
        time_coord_name : str, optional
            Name for time coordinate to use, by default 'time'
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
        _validate_time_coord(ds, time_coord_name)
        self._ds = ds
        self.attrs = {}

        self.attrs['is_time_decoded'] = _is_time_decoded(ds[time_coord_name])
        self.attrs['time_coord_name'] = time_coord_name
        self.attrs.update(self._get_time_attrs())
        self.attrs['use_cftime'] = _use_cftime(ds[time_coord_name], self.attrs)
        if self.attrs['is_time_decoded']:
            self.encoded_times = xr.coding.times.encode_cf_datetime(
                self._ds[self.attrs['time_coord_name']],
                units=self.attrs['units'],
                calendar=self.attrs['calendar'],
            )[0]

            self.encoded_time_bounds = xr.coding.times.encode_cf_datetime(
                self._ds[self.attrs['time_bounds_varname']],
                units=self.attrs['units'],
                calendar=self.attrs['calendar'],
            )[0]
        else:
            self.encoded_times = self._ds[self.attrs['time_coord_name']].copy()
            self.encoded_time_bounds = self._ds[self.attrs['time_bounds_varname']].copy()

        if binding is None:
            binding = _infer_time_data_tick_binding(
                self.encoded_time_bounds[0], self.encoded_times[0]
            )
            self.attrs['binding'] = binding
        elif binding in Axis._bindings:
            self.attrs['binding'] = binding
        else:
            message = f'Could not find the Time Axis binding associated to `{binding}`. '
            message += f'Possible options are: {list(Axis._bindings.keys())}'
            raise KeyError(message)

        self.decoded_times = xr.DataArray(
            dims=[self.attrs['time_coord_name']],
            data=xr.coding.times.decode_cf_datetime(
                Axis._bindings[self.attrs['binding']](
                    self.encoded_time_bounds, axis=self.attrs['time_bounds_dim_axis_num']
                ),
                self.attrs['units'],
                self.attrs['calendar'],
                use_cftime=self.attrs['use_cftime'],
            ),
        )

        dims = _get_time_bounds_dims(self.attrs)

        self.decoded_time_bounds = xr.DataArray(
            dims=dims,
            data=xr.coding.times.decode_cf_datetime(
                self.encoded_time_bounds,
                self.attrs['units'],
                self.attrs['calendar'],
                use_cftime=self.attrs['use_cftime'],
            ),
        )

    def _get_time_attrs(self):
        attrs = getattr(self._ds[self.attrs['time_coord_name']], 'attrs')
        encoding = getattr(self._ds[self.attrs['time_coord_name']], 'encoding')

        if 'units' in attrs:
            units = attrs['units']
        elif 'units' in encoding:
            units = encoding['units']

        if 'calendar' in attrs:
            calendar = attrs['calendar']
        elif 'calendar' in encoding:
            calendar = encoding['calendar']
        else:
            calendar = 'standard'

        if 'bounds' in attrs:
            time_bounds_varname = attrs['bounds']

        elif 'bounds' in encoding:
            time_bounds_varname = encoding['bounds']

        else:
            message = f'Unable to infer the time coordinate boundary variable'
            # TODO: Tell the user how to generate/provide time bounds.
            raise RuntimeError(message)

        dims = set(self._ds[time_bounds_varname].dims)
        time_coord_name = set([self.attrs['time_coord_name']])
        time_bounds_dim = (dims - time_coord_name).pop()
        time_bounds_dim_axis_num = self._ds[time_bounds_varname].get_axis_num(time_bounds_dim)
        return {
            'units': units,
            'calendar': calendar,
            'time_bounds_varname': time_bounds_varname,
            'time_bounds_dim': time_bounds_dim,
            'time_bounds_dim_axis_num': time_bounds_dim_axis_num,
        }


def _get_time_bounds_dims(attrs):

    if attrs['time_bounds_dim_axis_num'] == 1:
        dims = [attrs['time_coord_name'], attrs['time_bounds_dim']]
    else:
        dims = [attrs['time_bounds_dim'], attrs['time_coord_name']]

    return dims


def _validate_time_coord(ds, time_coord_name):
    if time_coord_name not in ds.variables:
        raise KeyError(f'Could not find time coordinate name = {time_coord_name} in the dataset.')


def _is_time_decoded(x):
    return xr.core.common._contains_datetime_like_objects(x)


def _use_cftime(x, attrs):
    if xr.core.common.contains_cftime_datetimes(x):
        return True

    elif xr.core.common.is_np_datetime_like(x.dtype):
        return False

    else:

        dummy = xr.coding.times.decode_cf_datetime(
            x[:2], units=attrs['units'], calendar=attrs['calendar']
        )
        if xr.core.common.is_np_datetime_like(np.array(dummy).dtype):
            return False
        else:
            return True


def _infer_time_data_tick_binding(sample_time_bound, sample_time_data_tick):

    if np.isclose(sample_time_data_tick, sample_time_bound[1]):
        return 'right'

    elif np.isclose(sample_time_data_tick, sample_time_bound[0]):
        return 'left'

    elif np.isclose(
        sample_time_data_tick,
        sample_time_bound[0] + (sample_time_bound[1] - sample_time_bound[0]) / 2.0,
    ):
        return 'middle'

    else:
        message = f"""Unable to infer time data tick binding from the input data set.
                   Please specify the `binding` parameter. Accepted values: {list(Axis._bindings.keys())}"""
        raise RuntimeError(message)
