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
experiments = ["historical", "ssp370", "ssp585"]


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
X370 = gmst_da["ssp370"].values.reshape(-1, 1)    
Y370 = lm.predict(X370)
Y370 = einops.rearrange(Y370, "t (lat lon) -> lat lon t", lat=96, lon=192)


# %%
cmip6_ssp370 = tas_da['ssp370']
emulator_ssp370 = utils.xarray_like(Y370, cmip6_ssp370)
cmip6_ssp370   = utils.wrap_lon(cmip6_ssp370)
emulator_ssp370 = utils.wrap_lon(emulator_ssp370)

year = slice(2030, 2060)
lon_range = slice(70, 120)
lat_range = slice(2, 32)

em_slice = emulator_ssp370.sel(year=year, lat=lat_range, lon=lon_range).mean('year')
cmip_slice = cmip6_ssp370.sel(year=year, lat=lat_range, lon=lon_range).mean('year')
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
plt.savefig("hiddenvar-ESM-pattern.pdf", dpi=300, bbox_inches='tight')
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
plt.savefig("hiddenvar-emulator-error.pdf", dpi=300, bbox_inches='tight')
plt.close()

# %%
