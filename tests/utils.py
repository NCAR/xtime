import numpy as np
import xarray as xr


def generate_time_and_bounds(bounds, fractions):
    """
    Generate time and time bounds

    Parameters
    ----------
    bounds : numpy.array
    fractions : numpy.array

    Returns
    -------
    ds : xarray.Dataset
    """
    time_bounds = np.vstack((bounds[:-1], bounds[1:])).T
    times = np.diff(time_bounds) * fractions.reshape(-1, 1) + time_bounds[:, 0].reshape(-1, 1)
    ds = xr.Dataset(
        {
            'time_bounds': xr.DataArray(
                time_bounds, dims=['time', 'd2'], coords={'time': times.flatten()}
            )
        }
    )
    return ds


def create_dataset(times, time_bounds, dtype='float32', nlats=2, nlons=2):
    """
    Create synthetic dataset
    """

    shape = times.shape[0], nlats, nlons

    a = np.random.randint(0, 10, shape).astype(dtype)
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')
    ds = xr.Dataset(
        {
            'x': xr.DataArray(
                a, dims=['time', 'lat', 'lon'], coords={'time': times, 'lon': lons, 'lat': lats}
            ),
            'time_bounds': xr.DataArray(time_bounds, dims=['time', 'd2']),
        }
    )
    return ds
