# %%
import os
import numpy as np
import einops
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.linear_model import LinearRegression
import utils


# %%
m = "MPI-ESM1-2-LR"
experiments = ["historical", "ssp119", "ssp585"]


# %%
tas_da = utils.load_data(m, "tas", experiments)
gmst_da = {e: utils.global_mean(tas_da[e]) for e in experiments}


# %%
X = np.concatenate([gmst_da["historical"].values, gmst_da["ssp585"].values])
Y = np.concatenate([tas_da["historical"].values, tas_da["ssp585"].values], axis=-1)
X = X.reshape(-1, 1)
Y = einops.rearrange(Y, "lat lon t -> t (lat lon)")
lm = LinearRegression().fit(X, Y)


# %%
lm = LinearRegression().fit(X, Y)
X119 = np.concatenate([gmst_da["ssp119"].values]).reshape(-1, 1)    
Y119 = lm.predict(X119)
Y119 = einops.rearrange(Y119, "t (lat lon) -> lat lon t", lat=96, lon=192)


# %%
cmip6_ssp119 = tas_da['ssp119']
emulator_ssp119 = utils.xarray_like(Y119, cmip6_ssp119)
cmip6_ssp119   = utils.wrap_lon(cmip6_ssp119)
emulator_ssp119 = utils.wrap_lon(emulator_ssp119)


# %%
year = 2100
lon_range = slice(-166, -40)
lat_range = slice(19, 90)

em_slice = emulator_ssp119.sel(year=year, lat=lat_range, lon=lon_range)
cmip_slice = cmip6_ssp119.sel(year=year, lat=lat_range, lon=lon_range)
em_error = em_slice - cmip_slice

all_vals = np.concatenate([cmip_slice.values.flatten(), em_error.values.flatten()])
vmax = np.nanquantile(np.abs(all_vals), 0.9999)

n_levels = 40
levels  = np.linspace(-vmax, +vmax, n_levels)




# %%
nrow, ncol = 1, 1
fig, ax = plt.subplots(nrow, ncol, figsize=(10 * ncol, 6 * nrow),
                       subplot_kw={'projection': ccrs.PlateCarree()})

cmip_slice.plot.contourf(ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap='bwr',
                        add_colorbar=False,
                        levels=levels,
                        vmax=vmax,
                        vmin=-vmax)
ax.coastlines()
ax.set_title("", fontsize=22)
ax.spines['geo'].set_visible(False)
plt.savefig("overshoot-ESM-pattern.pdf", dpi=300, bbox_inches='tight')
plt.close()


# %%
nrow, ncol = 1, 1
fig, ax = plt.subplots(nrow, ncol, figsize=(10 * ncol, 6 * nrow),
                       subplot_kw={'projection': ccrs.PlateCarree()})

em_error.plot.contourf(ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap='bwr',
                        add_colorbar=False,
                        levels=levels,
                        vmax=vmax,
                        vmin=-vmax)
ax.coastlines()
ax.set_title("", fontsize=22)
ax.spines['geo'].set_visible(False)
plt.savefig("overshoot-emulator-error.pdf", dpi=300, bbox_inches='tight')
plt.close()
# %%
