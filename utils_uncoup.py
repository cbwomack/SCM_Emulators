import numpy as np
import utils_general
from scipy.optimize import minimize

####################### Code to run uncoupled 3-box model #######################

########## Set Parameters ##########

# Grid parameters
dt = 1
Nt = 400 # Number of years
t = np.arange(0,dt*Nt,dt)
Nt = len(t)

# ODE parameters
rho_w = 1e3
c_w = 4e3
h = np.array([10.0, 150.0, 1500.0])
C = rho_w*c_w*h/3.154e7 # convert from seconds to years
lam = np.array([-0.86, -2.0, -0.67])
T0 = 0 # K
F0 = 1

# Forcing parameters
## 2xCO2 and 4xCO2 (constant forcing)
F_2xCO2 = utils_general.F_const(t, 3.7)
F_4xCO2 = utils_general.F_const(t, 7.4)

## High Warming
F_final = 0.4 # (W m^-2)
ts = 50
a_exp = F_final/np.exp(250*dt/ts)
F_exp = utils_general.F_exp(t, a_exp, ts)

## Overshoot
a_over = 2
b_over = 200
c_over = 70
F_over = utils_general.F_over(t, a_over, b_over, c_over)

## Impulse forcing
F0 = 1
F_pulse = utils_general.F_del(t, F0)
F_del = {'2xCO2':F_pulse,
         '4xCO2':F_pulse,
         'High Warming':F_pulse,
         'Overshoot':F_pulse}

## Compile all for diagnosis
F_all = {'2xCO2':F_2xCO2,
         '4xCO2':F_4xCO2,
         'High Warming':F_exp,
         'Overshoot':F_over}

# Plotting parameters
experiments = ['2xCO2','4xCO2','High Warming','Overshoot']
regions = ['Atmosphere', 'Shallow Ocean', 'Deep Ocean']
colors = utils_general.brewer2_light

########### Instantiate uncoupled model ##########

def create_uncoup(noisy=False):
  if noisy:
    pass
  else:
    T_ODE, g_ODE, a_ODE = timestep_uncoup(t, experiments, regions, F_all, lam, C, dt)
    utils_general.plot_box_model(T_ODE, t, experiments, regions, colors, soln_type='Noiseless ODE Solutions')

  return T_ODE, g_ODE, a_ODE

########## Methods for L(x,x') or R(x,t) ##########

def method_1_direct(T_ODE, modal=True):

  G, g, a = timestep_uncoup(t, experiments, regions, F_del, lam, C, dt)

  T_raw, T_modal = utils_general.estimate_T_1D(T_ODE, F_all,
                                               experiments, regions,
                                               t, C, 'G', G, a, g,
                                               T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 1 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L2_raw, L2_modal

def method_2_L(T_ODE, g_ODE, a_ODE, modal=True):

  t_range = (0,400)
  p = 1
  L_raw, L_modal = {}, {}
  for exp in experiments:
    L_raw[exp], L_modal[exp] = {}, {}
    for i in range(len(regions)):
      reg = regions[i]
      L_raw[exp][reg] = utils_general.calc_L_direct_1D(T_ODE[exp][reg], F_all[exp], t, t_range, C[i], p)
      L_modal[exp][reg] = utils_general.calc_L_direct_1D(-a_ODE[exp][reg][0], F_all[exp], t, t_range, C[i], p, modal=True, g=g_ODE[exp][reg])

  T_raw, T_modal = utils_general.estimate_T_1D(T_ODE, F_all,
                                               experiments, regions,
                                               t, C, 'L', L_raw, L_modal,
                                               None, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 2 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L2_raw, L2_modal

def method_3_deconv(T_ODE, modal=True):

  G, g, a = {}, {}, {}
  for exp in experiments:
    G[exp], g[exp], a[exp] = {}, {}, {}
    for i in range(len(regions)):
      reg = regions[i]
      G[exp][reg] = utils_general.calc_G_deconv_1D(T_ODE[exp][reg], F_all[exp], dt=1)
      g[exp][reg], a[exp][reg] = utils_general.calc_modes(G[exp][reg])

  T_raw, T_modal = utils_general.estimate_T_1D(T_ODE, F_all,
                                               experiments, regions,
                                               t, C, 'G', G, a,
                                               g, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 3 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L2_raw, L2_modal

def method_4_fit(T_ODE, g_ODE, a_ODE, modal=True):

  initial_guess = np.array([1, 1])
  options = {'disp': False}

  res_raw, res_modal = {}, {}
  G_fit, a_fit = {}, {}

  for exp in experiments:
    res_raw[exp], res_modal[exp] = {}, {}
    G_fit[exp], a_fit[exp] = {}, {}
    for reg in regions:
      res_raw[exp][reg] = minimize(utils_general.opt_h_lam_1D,
                                  initial_guess,
                                  args=(T_ODE[exp][reg], F_all[exp], t, 1),
                                  options=options)

      if modal:
        res_modal[exp][reg] = minimize(utils_general.opt_h_lam_1D,
                                      initial_guess,
                                      args=(a_ODE[exp][reg], F_all[exp], t, 1),
                                      options=options)

      G_fit[exp][reg] = utils_general.apply_response_1D(res_raw[exp][reg].x, t)
      a_fit[exp][reg] = utils_general.apply_response_1D(res_modal[exp][reg].x, t)

  T_raw, T_modal = utils_general.estimate_T_1D(T_ODE, F_all,
                                               experiments, regions,
                                               t, C, 'G', G_fit, a_fit,
                                               g_ODE, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 4 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L2_raw, L2_modal, res_raw

########### Helper functions ##########

def timestep_uncoup(t, experiments, regions, F, lam, C, dt, N_ensemble = None):

  T, g, a = {}, {}, {}
  for exp in experiments:
    T[exp], g[exp], a[exp] = {}, {}, {}
    for i in range(len(regions)):
      reg = regions[i]

      if N_ensemble is None:
        T_temp = np.zeros(len(t))
      else:
        T_temp = np.zeros((N_ensemble, len(t)))
        g_temp, a_temp = np.zeros((N_ensemble, 1)), np.zeros((N_ensemble, len(t)))

      for j in range(1, len(t)):
        if N_ensemble is None:
          T_temp[j] = T_temp[j-1] + (lam[i]/C[i] * T_temp[j-1] + F[exp][j-1]/C[i]) * dt
        else:
          T_temp[:,j] = T_temp[:,j-1] + (lam[i]/C[i] * T_temp[:,j-1] + F[exp][:,j-1]/C[i]) * dt

      if N_ensemble is not None:
        for n in range(N_ensemble):
          g_temp[n,:], a_temp[n, :] = utils_general.calc_modes(T_temp[n,:])
        g[exp][reg], a[exp][reg] = np.copy(g_temp), np.copy(a_temp)

      T[exp][reg] = np.copy(T_temp)

      if N_ensemble is None:
        g[exp][reg], a[exp][reg] = utils_general.calc_modes(T[exp][reg])

  return T, g, a

def calc_L2_and_plot(T_ODE, T_raw, t, experiments, regions, colors, soln_type, modal=True, T_modal=None):

  utils_general.plot_box_model(T_raw, t, experiments, regions, colors, soln_type=f'{soln_type} - Raw')
  L2_raw = utils_general.calc_L2(T_ODE, T_raw, experiments, regions, 'Raw')

  if modal:
    utils_general.plot_box_model(T_modal, t, experiments, regions, colors, soln_type=f'{soln_type} - Raw')
    L2_modal = utils_general.calc_L2(T_ODE, T_modal, experiments, regions, 'Modal')
    return L2_raw, L2_modal

  return L2_raw, None