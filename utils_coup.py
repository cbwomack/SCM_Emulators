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
M_a = 1
M_s = 5
M_d = 380
lam_as = 0.1
lam_sd = 0.1
gamma = 1/20
L = np.array([[-lam_as/M_a, lam_as/M_a, 0],
                [lam_as/M_s, -(lam_as + lam_sd)/M_s, lam_sd/M_s],
                [0, lam_sd/M_d, -lam_sd/M_d - gamma]])
T0 = 0 # K

# Forcing parameters
## 2xCO2 and 4xCO2 (constant forcing)
F_2xCO2 = utils_general.F_const(t, 0.05)
F_4xCO2 = utils_general.F_const(t, 0.1)

## High Emissions
F_final = 0.2 # (W m^-2)
ts = 50
a_exp = F_final/np.exp(400*dt/ts)
F_exp = utils_general.F_exp(t, a_exp, ts)

## Overshoot
a_over = 0.04
b_over = 200
c_over = 42.7
F_over = utils_general.F_over(t, a_over, b_over, c_over)

## Impulse forcing
F0 = 1
F_pulse = utils_general.F_del(t, F0)
F_del = {'2xCO2':F_pulse,
         '4xCO2':F_pulse,
         'High Emissions':F_pulse,
         'Overshoot':F_pulse}

## Compile all for diagnosis
F_all = {'2xCO2':F_2xCO2,
         '4xCO2':F_4xCO2,
         'High Emissions':F_exp,
         'Overshoot':F_over}

# Plotting parameters
experiments = ['2xCO2','4xCO2','High Emissions','Overshoot']
regions = ['Atmosphere', 'Shallow Ocean', 'Deep Ocean']
colors = utils_general.brewer2_light

########### Instantiate uncoupled model ##########

def create_coup(noisy=False):
  if noisy:
    pass
  else:
    T_ODE, g_ODE, a_ODE = timestep_coup(t, experiments, F_all, L, dt)
    utils_general.plot_box_model(T_ODE, t, experiments, regions, colors, soln_type='Noiseless ODE Solutions', coupled=True)

  return T_ODE, g_ODE, a_ODE

########## Methods for L(x,x') or R(x,t) ##########

def method_1_direct(T_ODE, modal=True):

  G_raw, g, a = timestep_coup(t, experiments, F_del, L, dt)
  G_modal = {}
  for exp in experiments:
    G_modal[exp] = g[exp] @ a[exp]

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'G',
                                               G_raw, G_modal, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 1 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, G_raw, G_modal

def method_2_L(T_ODE, g_ODE, a_ODE, modal=True):

  L_raw, L_modal = {}, {}
  for exp in experiments:
    L_raw[exp] = utils_general.calc_L_direct_2D(T_ODE[exp], F_all[exp], t)
    L_modal[exp] = utils_general.calc_L_direct_2D(a_ODE[exp], F_all[exp], t, modal=True, g=g_ODE[exp])

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'L',
                                               L_raw, L_modal,
                                               T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 2 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, L_raw, L_modal

def method_3_deconv(T_ODE, g_ODE, a_ODE, modal=True):

  G_raw, G_modal, a_deconv = {}, {}, {}
  for exp in experiments:
    G_raw[exp] = utils_general.calc_G_deconv_2D(T_ODE[exp], F_all[exp], dt=1)
    a_deconv[exp] = utils_general.calc_G_deconv_2D(a_ODE[exp], F_all[exp], dt=1)
    G_modal[exp] = g_ODE[exp] @ a_deconv[exp]

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'G',
                                               G_raw, G_modal, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 3 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, G_raw, G_modal

def method_4_fit(T_ODE, g_ODE, a_ODE, m, k, modal=True):

  gamma = np.ones(k)

  initial_v = np.random.rand(m * k)  # Flattened eigenvector
  initial_lam = np.random.rand(m)      # Eigenvalues
  initial_params = np.concatenate([initial_v, initial_lam])  # Combine into a single parameter vector
  bounds = [(None, None)] * (m * k) + [(-1, 0)] * m

  res_raw, res_modal = {}, {}
  G_raw, G_modal = {}, {}

  for exp in experiments:
    res_raw[exp] = minimize(utils_general.opt_v_lam_2D,
                            initial_params,
                            args=(T_ODE[exp], F_all[exp], t, m, dt, gamma),
                            method='L-BFGS-B',
                            bounds=bounds)

    res_modal[exp] = minimize(utils_general.opt_v_lam_2D,
                                  initial_params,
                                  args=(a_ODE[exp], F_all[exp], t, m, dt, gamma, g_ODE[exp]),
                                  method='L-BFGS-B',
                                  bounds=bounds)

    G_raw[exp] = utils_general.apply_v_lam_2D(res_raw[exp].x, t, m, gamma, dt)
    G_modal[exp] = utils_general.apply_v_lam_2D(res_modal[exp].x, t, m, gamma, dt, g_ODE[exp])

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'G',
                                               G_raw, G_modal, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments, regions,
                                      colors, soln_type='Method 4 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, G_raw, G_modal

########### Helper functions ##########

def timestep_coup(t, experiments, F, L, dt, gamma = None, N_ensemble = None):

  if gamma is None:
    gamma = np.array([1.0]*len(L))

  T, g, a = {}, {}, {}
  for exp in experiments:
    T[exp], g[exp], a[exp] = {}, {}, {}

    if N_ensemble is None:
      T_temp = np.zeros((3, len(t)))
    else:
      T_temp = np.zeros((N_ensemble, 3, len(t)))
      g_temp, a_temp = np.zeros((N_ensemble, 3, 3)), np.zeros((N_ensemble, 3, len(t)))

    for j in range(1, len(t)):
      if N_ensemble is None:
        T_temp[:,j] = T_temp[:,j-1] + (L @ T_temp[:,j-1] + gamma * F[exp][j-1]) * dt
      else:
        T_temp[:,:,j] = T_temp[:,:,j-1] + (T_temp[:, :, j-1] @ L.T + gamma[np.newaxis, :] * F[exp][:,j-1][:, np.newaxis]) * dt

      if N_ensemble is not None:
        for n in range(N_ensemble):
          g_temp[n,:], a_temp[n, :] = utils_general.calc_modes(T_temp[n,:,:])
        g[exp], a[exp] = np.copy(g_temp), np.copy(a_temp)

      T[exp] = np.copy(T_temp)

      if N_ensemble is None:
        g[exp], a[exp] = utils_general.calc_modes(T[exp])

  return T, g, a

def calc_L2_and_plot(T_ODE, T_raw, t, experiments, regions, colors, soln_type, modal=True, T_modal=None):

  utils_general.plot_box_model(T_raw, t, experiments, regions, colors, soln_type=f'{soln_type} - Raw', coupled=True)
  L2_raw = utils_general.calc_L2(T_ODE, T_raw, experiments, regions, 'Raw', coupled=True)

  if modal:
    utils_general.plot_box_model(T_modal, t, experiments, regions, colors, soln_type=f'{soln_type} - Modal', coupled=True)
    L2_modal = utils_general.calc_L2(T_ODE, T_modal, experiments, regions, 'Modal', coupled=True)
    return L2_raw, L2_modal

  return L2_raw, None