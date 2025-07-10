import os
from functools import singledispatch
import numpy as np
import xarray as xr


root_processed = "/Users/shahine/Documents/Postdoc/code/data/cmip6/processed"


def anomaly_file_path(model, experiment, variant, variable):
    return _write_anomaly_file_path(model, experiment, variant, variable, "Amon")


def write_cmip6_filename_prefix(model, experiment, variant, variable, table_id):
    prefix = f"{variable}_{table_id}_{model}_{experiment}_{variant}"
    return prefix


def _write_anomaly_file_path(model, experiment, variant, variable, table_id):
    # Generate the filename prefix
    prefix = write_cmip6_filename_prefix(model, experiment, variant, variable, table_id)

    # Construct the full filename and path
    filename = f"{prefix}_monthly_anomaly.nc"
    filepath = os.path.join(
        root_processed, model, experiment, variant, variable + "_anomaly", table_id, filename
    )
    return filepath


@singledispatch
def global_mean(data, *args, **kwargs):
    raise TypeError("Unsupported data type. Only xarray.DataArray and numpy.ndarray are supported.")


@global_mean.register
def _global_mean_datarray(data: xr.DataArray, lat_dim='lat', lon_dim='lon'):
    """
    Computes the global mean of a gridded DataArray
    """
    wlat = np.cos(np.deg2rad(data[lat_dim]))
    return data.weighted(wlat).mean((lat_dim, lon_dim))


@global_mean.register
def _global_mean_numpy(data: np.ndarray, lat, lat_dim=0, lon_dim=1):
    """
    Computes the global mean of a gridded numpy array with latitude as the first dimension
    """
    # Compute latitude weights
    wlat = np.cos(np.deg2rad(lat))
    wlat /= wlat.sum()

    # Expand weights to match the data shape (broadcast)
    shape = [1] * data.ndim
    shape[lat_dim] = len(lat) 
    wlat = wlat.reshape(shape)

    # Apply weights and compute global mean
    weighted_sum = np.sum(data * wlat, axis=(lat_dim, lon_dim))
    global_mean = weighted_sum / data.shape[lon_dim]
    return global_mean


def groupby_month_and_year(ds):
    """
    Reshapes datarray time dimension to group by month and year
    """
    year = ds.time.dt.year
    month = ds.time.dt.month
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
    return ds.set_index(time=("year", "month")).unstack("time")


def xarray_like(numpy_array, example_xr_dataarray):
    """
    Create a new DataArray from a numpy array with the same dimensions and coordinates as an existing DataArray,
    """
    # Ensure the input is a NumPy array
    numpy_array = np.asarray(numpy_array)

    # Check if the shape matches
    if numpy_array.shape != example_xr_dataarray.shape:
        raise ValueError(f"Shape mismatch: array shape {numpy_array.shape} does not match "
                         f"xr_dataarray shape {example_xr_dataarray.shape}.")

    # Create the new DataArray
    field = xr.DataArray(
        data=numpy_array,
        dims=example_xr_dataarray.dims,
        coords=example_xr_dataarray.coords,
    )
    return field


def wrap_lon(ds):
    # assumes ds.lon runs 0…360
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds


def load_data(m, v, experiments):
    var_datasets = dict()
    for e in experiments:
        variants = os.listdir(os.path.join(root_processed, m, e))
        variants.sort()
        filepaths = [anomaly_file_path(m, e, ω, v) for ω in variants]
        filepaths = filter(os.path.exists, filepaths)
        realizations = [
                xr.open_dataset(fp).assign_coords(variant=v, experiment=e)
                for fp, v in zip(filepaths, variants)
            ]
        var_ds = xr.concat(realizations, dim="member")[v].mean('member')
        var_ds = groupby_month_and_year(var_ds).mean('month')
        var_datasets[e] = var_ds
    return var_datasets
