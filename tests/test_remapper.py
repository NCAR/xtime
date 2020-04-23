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
        incoming_axis_coord=incoming.time.data,
        outgoing_axis_coord=outgoing.time.data,
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
        incoming_axis_coord=incoming.time.data,
        outgoing_axis_coord=outgoing.time.data,
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
        incoming_axis_coord=incoming.time.data,
        outgoing_axis_coord=outgoing.time.data,
    )
    ds = dataset.chunk()
    remapped_data = remapper(ds.x)
    assert isinstance(remapped_data.data, dask.array.Array)
    assert remapped_data.shape == (24, 2, 2)
    assert set(remapped_data.dims) == set(dataset.x.dims)
    xr.testing.assert_equal(remapped_data, remapper(dataset.x))
