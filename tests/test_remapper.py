import dask.array
import numpy as np
import pytest
import sparse
import xarray as xr

from xgriddedaxis import Remapper
from xgriddedaxis.remapper import _FROM_KEY, _TO_KEY

from .utils import create_dataset, generate_time_and_bounds


@pytest.fixture(scope='module')
def from_axis():
    n = 13
    bounds = np.round(np.logspace(2.0, 3.5, num=n), decimals=0)
    fractions = np.round(np.random.random(n - 1), decimals=3)
    from_axis = generate_time_and_bounds(bounds, fractions)
    return from_axis


@pytest.fixture(scope='module')
def to_axis():
    n = 25
    bounds = np.round(np.logspace(2.0, 3.6, num=n), decimals=0)
    fractions = np.round(np.random.random(n - 1), decimals=3)
    to_axis = generate_time_and_bounds(bounds, fractions)
    return to_axis


@pytest.fixture(scope='module')
def dataset(from_axis, to_axis):
    ds = create_dataset(from_axis['time'], from_axis['time_bounds'])
    return ds


def test_remapper_init(from_axis, to_axis):
    remapper = Remapper(
        from_axis=from_axis, to_axis=to_axis, axis_name='time', boundary_variable='time_bounds',
    )
    assert isinstance(remapper.weights.data, sparse._coo.core.COO)
    assert remapper.weights.shape == (
        remapper.to_axis[remapper.axis_name].size,
        remapper.from_axis[remapper.axis_name].size,
    )
    assert set(remapper.weights.dims) == set([_TO_KEY, _FROM_KEY])


def test_remapper_apply_weights(dataset, from_axis, to_axis):
    remapper = Remapper(
        from_axis=from_axis, to_axis=to_axis, axis_name='time', boundary_variable='time_bounds',
    )
    remapped_data = remapper(dataset.x)
    assert remapped_data.shape == (24, 2, 2)
    assert set(remapped_data.dims) == set(dataset.x.dims)


def test_remapper_apply_weights_dask(dataset, from_axis, to_axis):
    remapper = Remapper(
        from_axis=from_axis, to_axis=to_axis, axis_name='time', boundary_variable='time_bounds',
    )
    ds = dataset.chunk()
    remapped_data = remapper(ds.x)
    assert isinstance(remapped_data.data, dask.array.Array)
    assert remapped_data.shape == (24, 2, 2)
    assert set(remapped_data.dims) == set(dataset.x.dims)
    xr.testing.assert_equal(remapped_data, remapper(dataset.x))


def test_remapper_apply_weights_invalid_input(dataset, from_axis, to_axis):
    remapper = Remapper(
        from_axis=from_axis, to_axis=to_axis, axis_name='time', boundary_variable='time_bounds',
    )

    with pytest.raises(NotImplementedError):
        _ = remapper(dataset)

    with pytest.raises(TypeError):
        _ = remapper(dataset.x.data)


def test_bounds_sanity_check():
    from xgriddedaxis.remapper import _bounds_sanity_check

    bounds = np.array([0.0, 15.0, 10.0, 22.0, 26.0, 50.0, 45.0])
    fractions = np.array([0.5, 0.4, 0.2, 1, 0.5, 0.3])
    from_axis = generate_time_and_bounds(bounds, fractions)
    with pytest.raises(ValueError, match=r'all lower bounds must be smaller'):
        _bounds_sanity_check(from_axis.time_bounds)

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

    x = np.array([1.0, 1.0])
    with pytest.raises(AssertionError, match=r'data ticks must be monotically increasing'):
        _data_ticks_sanity_check(x)
