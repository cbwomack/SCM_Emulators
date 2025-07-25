import xarray as xr
import numpy as np


time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)


MPI = xr.open_mfdataset("/home/filepath/to/MPI-ESM1-2-LR/piControl/r1i1p1f1/tas/Amon/*", decode_times=time_coder).drop_vars('height')
wlat = np.cos(np.deg2rad(MPI.lat))
global_annual_MPI = MPI['tas'].weighted(wlat).mean(['lat', 'lon']).groupby('time.year').mean()
σMPI = global_annual_MPI.std().compute()  # 0.11147713


MIROC = xr.open_mfdataset("/home/filepath/to/MIROC6/piControl/r1i1p1f1/tas/Amon/*", decode_times=time_coder).drop_vars('height')
wlat = np.cos(np.deg2rad(MIROC.lat))
global_annual_MIROC = MIROC['tas'].weighted(wlat).mean(['lat', 'lon']).groupby('time.year').mean()
σMIROC = global_annual_MIROC.std().compute()  # 0.12811273


ACCESS = xr.open_mfdataset("/home/filepath/to/ACCESS-ESM1-5/piControl/r1i1p1f1/tas/Amon/*", decode_times=time_coder).drop_vars('height')
wlat = np.cos(np.deg2rad(ACCESS.lat))
global_annual_ACCESS = ACCESS['tas'].weighted(wlat).mean(['lat', 'lon']).groupby('time.year').mean()
σACCESS = global_annual_ACCESS.std().compute()  # 0.11018653


σ = (σMPI + σMIROC + σACCESS) / 3   # 0.11659213