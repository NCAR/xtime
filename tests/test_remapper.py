import dask.array
import numpy as np
import pytest
import sparse
import xarray as xr

from xgriddedaxis import Remapper
from xgriddedaxis.remapper import _INCOMING_KEY, _OUTGOING_KEY

from .utils import create_dataset, generate_time_and_bounds


@pytest.fixture(scope='module')
def incoming():
    n = 13
    bounds = np.round(np.logspace(2.0, 3.5, num=n), decimals=0)
    fractions = np.round(np.random.random(n - 1), decimals=1)
    incoming = generate_time_and_bounds(bounds, fractions)
    return incoming


@pytest.fixture(scope='module')
def outgoing():
    n = 25
    bounds = np.round(np.logspace(2.0, 3.6, num=n), decimals=0)
    fractions = np.round(np.random.random(n - 1), decimals=1)
    outgoing = generate_time_and_bounds(bounds, fractions)
    return outgoing


@pytest.fixture(scope='module')
def dataset(incoming, outgoing):
    ds = create_dataset(incoming['time'], incoming['time_bounds'])
    return ds


def test_remapper_init(incoming, outgoing):
    remapper = Remapper(
        incoming_axis=incoming,
        outgoing_axis=outgoing,
        axis_name='time',
        boundary_variable='time_bounds',
    )
    assert isinstance(remapper.weights.data, sparse._coo.core.COO)
    assert remapper.weights.shape == (
        remapper.outgoing_axis[remapper.axis_name].size,
        remapper.incoming_axis[remapper.axis_name].size,
    )
    assert set(remapper.weights.dims) == set([_OUTGOING_KEY, _INCOMING_KEY])


def test_remapper_apply_weights(dataset, incoming, outgoing):
    remapper = Remapper(
        incoming_axis=incoming,
        outgoing_axis=outgoing,
        axis_name='time',
        boundary_variable='time_bounds',
    )
    remapped_data = remapper(dataset.x)
    assert remapped_data.shape == (24, 2, 2)
    assert set(remapped_data.dims) == set(dataset.x.dims)


def test_remapper_apply_weights_dask(dataset, incoming, outgoing):
    remapper = Remapper(
        incoming_axis=incoming,
        outgoing_axis=outgoing,
        axis_name='time',
        boundary_variable='time_bounds',
    )
    ds = dataset.chunk()
    remapped_data = remapper(ds.x)
    assert isinstance(remapped_data.data, dask.array.Array)
    assert remapped_data.shape == (24, 2, 2)
    assert set(remapped_data.dims) == set(dataset.x.dims)
    xr.testing.assert_equal(remapped_data, remapper(dataset.x))


def test_remapper_apply_weights_invalid_input(dataset, incoming, outgoing):
    remapper = Remapper(
        incoming_axis=incoming,
        outgoing_axis=outgoing,
        axis_name='time',
        boundary_variable='time_bounds',
    )

    with pytest.raises(NotImplementedError):
        _ = remapper(dataset)

    with pytest.raises(TypeError):
        _ = remapper(dataset.x.data)


def test_bounds_sanity_check():
    from xgriddedaxis.remapper import _bounds_sanity_check

    bounds = np.array([0.0, 15.0, 10.0, 22.0, 26.0, 50.0, 45.0])
    fractions = np.array([0.5, 0.4, 0.2, 1, 0.5, 0.3])
    incoming = generate_time_and_bounds(bounds, fractions)
    with pytest.raises(ValueError, match=r'all lower bounds must be smaller'):
        _bounds_sanity_check(incoming.time_bounds)

    bounds = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(AssertionError, match=r'Bounds must be a 2D array.'):
        _bounds_sanity_check(bounds)

    bounds = bounds.reshape(2, 4)
    with pytest.raises(AssertionError, match=r'Bounds must be a 2D array with shape'):
        _bounds_sanity_check(bounds)


def test_data_ticks_sanity_check():
    from xgriddedaxis.remapper import _data_ticks_sanity_check

    x = np.arange(4)
    _data_ticks_sanity_check(x)

    x = np.arange(4).reshape(4, 1)
    with pytest.raises(AssertionError, match=r'data ticks must be a one dimensional array'):
        _data_ticks_sanity_check(x)

    x = np.array([10.0, 13.0, 15.0, 14.0, 20.0])
    with pytest.raises(AssertionError, match=r'data ticks must be monotically increasing'):
        _data_ticks_sanity_check(x)
