# %%
import numpy as np
import einops
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import utils



# %%
m = "MPI-ESM1-2-LR"
experiments = ["historical", "ssp585"]


# %%
tas_da = utils.load_data(m, "tas", experiments)
gmst_da = {e: utils.global_mean(tas_da[e]) for e in experiments}


# %%
X = gmst_da["historical"].values
poly = PolynomialFeatures(degree=12, include_bias=False)
X = poly.fit_transform(X.reshape(-1, 1))

# %%
Y = tas_da["historical"].values
Y = einops.rearrange(Y, "lat lon t -> t (lat lon)")
lm = LinearRegression().fit(X, Y)


# %%
lm = LinearRegression().fit(X, Y)
X585 = poly.transform(gmst_da["ssp585"].values.reshape(-1, 1))
Y585 = lm.predict(X585)
Y585 = einops.rearrange(Y585, "t (lat lon) -> lat lon t", lat=96, lon=192)


# %%
cmip6_ssp585 = tas_da['ssp585']
emulator_ssp585 = utils.xarray_like(Y585, cmip6_ssp585)
cmip6_ssp585   = utils.wrap_lon(cmip6_ssp585)
emulator_ssp585 = utils.wrap_lon(emulator_ssp585)

year = slice(2020, 2020)
lon_range = slice(130, 280)
lat_range = slice(-41, 41)

em_slice = emulator_ssp585.sel(year=year, lat=lat_range, lon=lon_range).mean('year')
cmip_slice = cmip6_ssp585.sel(year=year, lat=lat_range, lon=lon_range).mean('year')
em_error = em_slice - cmip_slice

all_vals = np.concatenate([cmip_slice.values.flatten(), em_error.values.flatten()])
vmax = np.nanquantile(np.abs(all_vals), 0.999)

n_levels = 40
levels  = np.linspace(-vmax, +vmax, n_levels)


# %%
nrow, ncol = 1, 1
fig, ax = plt.subplots(nrow, ncol, figsize=(10 * ncol, 6 * nrow),
                       subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180.0)})

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
plt.savefig("noise-ESM-pattern.pdf", dpi=300, bbox_inches='tight')
plt.close()


# %%
nrow, ncol = 1, 1
fig, ax = plt.subplots(nrow, ncol, figsize=(10 * ncol, 6 * nrow),
                       subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180.0)})

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
plt.savefig("noise-emulator-error.pdf", dpi=300, bbox_inches='tight')
plt.close()
# %%
