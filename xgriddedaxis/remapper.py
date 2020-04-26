import dask
import numpy as np
import sparse
import xarray as xr
from scipy.sparse import csr_matrix

_OUTGOING_KEY = 'outgoing'
_INCOMING_KEY = 'incoming'


class Remapper:
    """
    An object that facilitates conversion of data between two one dimensional axes.
    """

    def __init__(
        self, incoming_axis, outgoing_axis, axis_name, boundary_variable,
    ):
        """
        Create a new Remapper object that facilitates conversion of data between two
        one-dimensional axes.

        Parameters
        ----------
        incoming_axis : xarray.Dataset
           Contains incoming/from axis information. This dataset should have
           a 2D bounds variable (containing lower and upper bounds) and the
           corresponding data ticks as a coordinate. The data tick defines where
           in the bounds interval you are associating the data point.
        outgoing_axis : xarray.Dataset
           Contains outgoing/to axis information. This dataset should have
           a 2D bounds variable (containing lower and upper bounds) and the
           corresponding data ticks as a coordinate. The data tick defines where
           in the bounds interval you are associating the data point.
        axis_name : str
           Name of the axis. For example, `time`, `lat`, etc..
        boundary_variable : str
           Name of the variable with bounds array.

        Notes
        -----
        The lower bounds values must be monotonically increasing.
        The upper bounds do NOT need to be monotonically increasing.
        However, they should not be smaller than their counter-part lower bounds.

        """
        self.axis_name = axis_name
        self.boundary_variable = boundary_variable
        self.incoming_axis = incoming_axis
        self.outgoing_axis = outgoing_axis
        self.coverage_info = None
        self.weights = self.generate_weights()

    def generate_weights(self):
        """
        Generate remapping weights.
        """
        self.coverage_info = get_coverage_info(
            self.incoming_axis[self.boundary_variable].data,
            self.outgoing_axis[self.boundary_variable].data,
        )

        incoming_data_ticks = self.incoming_axis[self.axis_name].data
        outgoing_data_ticks = self.outgoing_axis[self.axis_name].data

        _data_ticks_sanity_check(incoming_data_ticks)
        _data_ticks_sanity_check(outgoing_data_ticks)
        coords = {
            _INCOMING_KEY: incoming_data_ticks,
            _OUTGOING_KEY: outgoing_data_ticks,
        }
        weights = construct_coverage_matrix(
            self.coverage_info['weights'],
            self.coverage_info['col_idx'],
            self.coverage_info['row_idx'],
            self.coverage_info['shape'],
            coords=coords,
        )
        return weights

    def __call__(self, data):
        """
        Apply remapping weights to data.

        Parameters
        ----------
        data : xarray.DataArray, xarray.Dataset
            Data to map from the incoming axis to the outgoing axis.
        Returns
        -------
        outdata : xarray.DataArray, xarray.Dataset
            Remapped data. Data type is the same as input data type.
            All the dimensions are the same as the input data except the incoming axis.

        Raises
        ------
        TypeError
            if input data is not an xarray DataArray or xarray Dataset.
        """
        if isinstance(data, xr.DataArray):
            return self._remap_dataarray(data)
        elif isinstance(data, xr.Dataset):
            return self._remap_dataset(data)
        else:
            raise TypeError('input data must be xarray DataArray or xarray Dataset!')

    def _remap_dataarray(self, dr_in):
        weights = self.weights.copy()
        # Convert sparse matrix into a dense one to avoid TypeError when performing dot product
        # TypeError: no implementation found for 'numpy.einsum' on types that implement
        # __array_function__: [<class 'sparse._coo.core.COO'>, <class 'numpy.ndarray'>]
        weights.data = weights.data.todense()
        indata = _sanitize_input_data(dr_in, self.axis_name, self.weights)
        if isinstance(indata.data, dask.array.Array):
            incoming_axis_chunks = dict(zip(indata.dims, indata.chunks))[self.axis_name][0]

            weights = weights.chunk({_OUTGOING_KEY: incoming_axis_chunks})
            return _apply_weights(weights, indata, self.axis_name,)
        else:
            return _apply_weights(weights, indata, self.axis_name)

    def _remap_dataset(self, ds_in):
        raise NotImplementedError('Currently only works on xarray DataArrays')


def _sanitize_input_data(data, axis_name, weights):
    message = (
        f'The length ({data[axis_name].size}) of incoming {axis_name} dimension does not match '
        f"with the provided remapper object's incoming {axis_name} dimension ({weights[_INCOMING_KEY].size})"
    )
    assert data[axis_name].size == weights[_INCOMING_KEY].size, message
    indata = data.copy()
    indata[axis_name] = weights[_INCOMING_KEY].data
    return indata


def _apply_weights(weights, indata, axis_name):
    """
    Apply remapping weights to data. We apply weights normalization
    when we have missing values in the input data.

    Parameters
    ----------
    weights : xarray.DataArray
       Remapping weights
    indata : xarray.DataArray
       Input data to remap to a new axis.
    axis_name : str
        Name of the axis. For example, `time`, `lat`, etc..

    Returns
    -------
    outdata : xarray.DataArray
       Remapped data

    """
    indata = indata.rename({axis_name: _INCOMING_KEY})
    nan_mask = indata.isnull()
    non_nan_mask = xr.ones_like(indata, dtype=np.int8)
    non_nan_mask = non_nan_mask.where(~nan_mask, 0)
    indata = indata.where(~nan_mask, 0)
    inverse_sum_effective_weights = np.reciprocal(xr.dot(weights, non_nan_mask))
    outdata = xr.dot(weights, indata) * inverse_sum_effective_weights
    return outdata.rename({_OUTGOING_KEY: axis_name})


def get_coverage_info(incoming_bounds, outgoing_bounds):
    """
    Compute the overlap/coverage between the incoming and outgoing bounds

    Parameters
    ----------
    incoming_bounds : numpy.array
        incoming bounds
    outgoing_bounds : numpy.array
        outgoing bounds
    Returns
    -------
    dict
        Dictionary containing information used to generate remapping weights matrix
    """
    _bounds_sanity_check(incoming_bounds)
    _bounds_sanity_check(outgoing_bounds)

    incoming_lower_bounds = incoming_bounds[:, 0]
    incoming_upper_bounds = incoming_bounds[:, 1]
    outgoing_lower_bounds = outgoing_bounds[:, 0]
    outgoing_upper_bounds = outgoing_bounds[:, 1]

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

    coverage = {
        'weights': weights,
        'col_idx': col_idx,
        'row_idx': row_idx,
        'shape': (m, n),
    }
    return coverage


def construct_coverage_matrix(weights, col_idx, row_idx, shape, coords):
    """
    Generate remapping weights sparse matrix.

    Parameters
    ----------
    weights : array_like
        Contain overlap/coverage between the incoming and outgoing bounds
    col_idx : array_like
        column indices
    row_idx : array_like
        row indices
    shape : tuple
        Shape of the matrix
    coords : dict
        Dictionary-like container of coordinate arrays.

    Returns
    -------
    xarray.DataArray
        Contains the remapping weights (stored as a sparse matrix in COO format)
    """
    wgts = csr_matrix((weights, (row_idx, col_idx)), shape=shape).tolil()
    mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
    wgts[mask, 0] = np.nan
    wgts = sparse.COO.from_scipy_sparse(wgts)
    weights = xr.DataArray(data=wgts, dims=['outgoing', 'incoming'], coords=coords)
    return weights


def _bounds_sanity_check(bounds):
    # Make sure lower_i <= upper_i
    if bounds.shape[1] > 1:
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(
                'all lower bounds must be smaller than their counter-part upper bounds'
            )

        # Make sure lower_i < lower_{i+1}
        if np.any(bounds[0, :-1] >= bounds[0, 1:]):
            raise ValueError('lower bound values must be monotonically increasing.')


def _data_ticks_sanity_check(data_ticks):
    message = 'data ticks must be monotically increasing.'
    assert np.any(data_ticks[:-1] <= data_ticks[1:]), message
