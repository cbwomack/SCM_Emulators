import utils_general
import xarray as xr
import numpy as np

def method_2_L(T_ODE, g_ODE, a_ODE, experiments, modal=True, global_mean=True):

  L_raw, L_modal = {}, {}
  for exp in experiments:
    L_raw[exp] = utils_general.calc_L_direct_2D(T_ODE[exp], F_all[exp], t)
    L_modal[exp] = utils_general.calc_L_direct_2D(a_ODE[exp], F_all[exp], t, modal=True, g=g_ODE[exp])

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                              experiments, t, 'L',
                                              L_raw, L_modal,
                                              T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments,
                                      soln_type='Method 2 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L_raw, L_modal

############# Data importing ###########

def import_calc_climatology(piCtrl_ds_path, var, global_mean=True):

  piCtrl_ds = xr.open_mfdataset(piCtrl_ds_path)
  piCtrl_annual_ds = monthly_to_annual(piCtrl_ds, var)
  climatology_ds = calc_climatology(piCtrl_annual_ds)

  if global_mean:
    global_climatology_ds = calc_global_mean(climatology_ds, var)

    return global_climatology_ds

  return climatology_ds

def import_annual(ds_path, var, global_mean=True):

  ds = xr.open_mfdataset(ds_path)
  annual_ds = monthly_to_annual(ds, var)

  if global_mean:
    global_ds = calc_global_mean(annual_ds, var)

    return global_ds

  return annual_ds

def monthly_to_annual(ds, var):

  days_in_month = ds.time.dt.days_in_month
  weights = days_in_month.groupby("time.year") / days_in_month.groupby("time.year").sum()

  var_ds = ds[var]
  cond = var_ds.isnull()
  ones = xr.where(cond, 0.0, 1.0)
  var_sum = (var_ds * weights).resample(time="YS").sum(dim="time")
  ones_sum = (ones * weights).resample(time="YS").sum(dim="time")
  annual_ds = var_sum / ones_sum

  years = annual_ds["time.year"].data
  annual_ds = annual_ds.assign_coords(time=("time", years))

  return annual_ds

def calc_climatology(annual_ds):
  climatology_ds = annual_ds.mean(dim = ['time'])
  return climatology_ds.as_numpy()

def calc_global_mean(ds, var):

  weights = np.cos(np.deg2rad(ds.lat))
  weights.name = "weights"
  weighted = ds.weighted(weights)
  weighted_mean = weighted.mean(['lat','lon'])

  return weighted_mean.as_numpy()

def calc_energy_balance(rlut, rsdt, rsut):

    longwave_balance = -rlut
    shortwave_balance = rsdt - rsut
    dN = (longwave_balance + shortwave_balance).to_dataset(name = 'RF')

    return dN

def calc_ERF(tas_global, rlut, rsdt, rsut, lam):

  dN = calc_energy_balance(rlut, rsdt, rsut)
  ERF_ds = dN['RF'] + lam*tas_global

  return ERF_ds