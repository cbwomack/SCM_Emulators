from curses import killchar
from fcntl import F_DUPFD
from multiprocessing.sharedctypes import Value
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse, convolve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize
from scipy.special import erf
from scipy.linalg import toeplitz
from scipy import sparse

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D



#################### General Helper Functions ####################

##### Forcing profiles
# No forcing
def F_none(t):
  return np.zeros(len(t))

# Constant forcing
def F_const(t, F0):
  return F0*np.ones(len(t))

# Exponential forcing
def F_exp(t, a, ts):
  return a*np.exp(t/ts)

# Delta forcing
def F_del(t, F0):
  return F0*unit_impulse(len(t))

# Overshoot forcing
def F_over(t, a, b, c):
  return a*np.exp(-np.square(t - b)/(2*c**2))

##### Solutions given the above forcing profiles
# General ODE 1D: C*dT(t)/dt = -lam*T(t) + F(t)

# F(t) = 0
def T_none(t, T0, k, dim=1, x=None, y=None):
  if dim == 1:
    return T0*np.exp(-k*t)
  elif dim == 2:
    if x is not None:
      return T0*np.exp(-k*x*t)
    else:
      raise ValueError("For dim=2, x must be provided.")
  elif dim == 3:
    if x is not None and y is not None:
      xy = np.outer(x, y).reshape(-1, 1)
      return T0 * np.exp(-k * xy * t)
    else:
      raise ValueError("For dim=3, both x and y must be provided.")
  else:
    raise ValueError("dim must be 1, 2, or 3.")

# F(t) = F
def T_const(t, T0, lam, F0, C, dim=1):
  if dim == 1:
    return (T0*lam*np.exp(t*(lam/C)) + F0*np.exp(t*(lam/C)) - F0)/lam
  else:
    raise ValueError("dim must be 1.")

# F(t) = a*exp(t/ts)
def T_exp(t, ts, a, T0, lam, C, dim=1):
  if dim == 1:
    return (np.exp(t*(lam/C))*(C*T0 + a*ts*(np.exp(t*(1/ts - lam/C))) - T0*ts*lam))/(C - ts*lam)
  else:
    raise ValueError("dim must be 1.")

# F(t) = a*exp(-(t - b)^2/(2*c^2))
def T_over(t, T0, a, b, c, Lambda, C, dim=1):
  if dim == 1:
    term1 = 2 * C * T0

    exp_term = np.exp((Lambda * (-2 * b * C + c**2 * Lambda)) / (2 * C**2))
    sqrt_term = np.sqrt(2 * np.pi)

    erf1 = erf((-b * C + c**2 * Lambda) / (np.sqrt(2) * c * C))
    erf2 = erf((-b * C + C * t + c**2 * Lambda) / (np.sqrt(2) * c * C))

    term2 = a * c * exp_term * sqrt_term * (erf1 - erf2)

    T = np.exp(t*Lambda/C)*(term1 - term2) / (2 * C)

    return T

  else:
    raise ValueError("dim must be 1.")

# F(t) = F*delta(t)
def T_del(t, lam, F0, C, dt, dim=1):
  if dim == 1:
    T_d = np.roll(np.exp(t*(lam/C))*(np.heaviside(t, F0))/(C/dt), 1)
    #T_d = np.exp(t*(lam/C))*(np.heaviside(t, F0))/(C/dt)
    T_d[0] = 0
    return T_d
  else:
    raise ValueError("dim must be 1")

def calc_temp_response_1D(exp, t, T0, F_2xCO2, F_4xCO2, ts, a_exp, a_over, b_over, c_over, lam, C):
  if exp == '2xCO2': return T_const(t, T0, lam, F_2xCO2, C)
  elif exp == '4xCO2': return T_const(t, T0, lam, F_4xCO2, C)
  elif exp == 'RCP70': return T_exp(t, ts, a_exp, T0, lam, C)
  elif exp == 'Overshoot': return T_over(t, T0, a_over, b_over, c_over, lam, C)
  else:
    raise ValueError(f"exp: {exp} not found")

##### Solver Methodologies

# Reconstruct temperature profile given a linear operator
def reconstruct_T(F, C, t, L, T0, dt):
  N_t = len(t)
  T_G = np.zeros(N_t)
  T_G[0] = T0

  for i in range(1, N_t):
    T_G[i] = dt*(L[0][0] + 1)*T_G[i-1] + dt*F[i-1]/C

  return T_G

##### Plotting Functions

#################### 1D Helper Functions ####################

def calc_modes_1D(T, t):
    g, s, Vh = linalg.svd(T.reshape(1,len(t)), full_matrices=False)
    a = s*Vh
    return g, a

def calc_L_direct_1D(T, F, t, t_range, C, p_order, modal=False, g=None):

  T = T[t_range[0]:t_range[1]]
  F = F[t_range[0]:t_range[1]]
  t = t[t_range[0]:t_range[1]]

  R = np.copy(T).reshape(1,len(t))
  dR = finite_difference_derivative(T, t, p_order)
  dR_dt = np.copy(dR).reshape(1,len(t))
  #dR_dt = calc_derivative(R, t)
  R_p = linalg.pinv(R)

  if modal:
    if np.shape(g) != (1,1):
      g = g.reshape(1,1)
    N = np.dot(np.subtract(dR_dt, np.divide(F,C)), R_p)
    L = g[0]*N*linalg.pinv(g)[0]
  else:
    L = np.dot(np.subtract(dR_dt, np.divide(F,C)), R_p)

  return L

def estimate_T_1D(T_analytic, F_all, experiments, regions, t, C, mode, op_raw, op_modal, g, T0, dt):
  T_est_raw, T_est_modal = {}, {}
  n_exp = len(experiments)
  n_reg = len(regions)

  # Iterate over forcing profiles
  for exp in experiments:
    T_est_raw[exp], T_est_modal[exp] = {}, {}
    # Iterate over regions
    for i in range(n_reg):
      reg = regions[i]
      if mode == 'L':
        T_est_raw[exp][reg] = reconstruct_T(F_all[exp], C[i], t, op_raw[exp][reg], T0, dt)
        T_est_modal[exp][reg] = reconstruct_T(F_all[exp], C[i], t, op_modal[exp][reg], T0, dt)
      elif mode == 'G':
        T_est_raw[exp][reg] = convolve(op_raw[exp][reg]*dt, F_all[exp])[:len(t)]
        if len(op_modal[exp][reg]) == 1:
          T_est_modal[exp][reg] = g[exp][reg][0]*convolve(op_modal[exp][reg][0]*dt, F_all[exp])[:len(t)]
        else:
          T_est_modal[exp][reg] = g[exp][reg][0]*convolve(op_modal[exp][reg]*dt, F_all[exp])[:len(t)]

  return T_est_raw, T_est_modal

def plot_uncoupled(T, t, experiments, regions, colors, soln_type, ensemble=False):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(1, n_exp, figsize=(5*n_exp,5), sharey=True)

  for i in range(n_exp):
    exp = experiments[i]
    for j in range(n_reg):
      reg = regions[j]
      if ensemble:
        T_mean = np.mean(T[exp][reg], axis=0)
        T_std = np.std(T[exp][reg], axis=0)

        ax[i].plot(t, T_mean, c=colors[j], label='Ensemble Mean', linewidth=2)
        ax[i].fill_between(t, T_mean - T_std, T_mean + T_std, color=colors[j], alpha=0.5, label='Ensemble Spread (±1 std)')
        ax[i].fill_between(t, T_mean - 2*T_std, T_mean + 2*T_std, color=colors[j], alpha=0.2, label='Ensemble Spread (±2 std)')

      else:
        ax[i].plot(t, T[exp][reg], c=colors[j], label=regions[j], lw=2)

    ax[i].set_title(f'{soln_type} - {experiments[i]}')
    ax[i].set_xlabel('Year')
    #ax[i].legend()

  ax[0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  plt.tight_layout()

  return

def plot_box_model(T, t, experiments, regions, colors, soln_type, coupled=False, ensemble=False):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(1, n_exp, figsize=(5*n_exp,5))#, sharey=True, constrained_layout=True)

  for i, exp in enumerate(experiments):
    for j, reg in enumerate(regions):
      if ensemble is False:
        if coupled:
          T_temp = np.array(T[exp])[j,:]
        else:
          T_temp = T[exp][reg]
        ax[i].plot(t, T_temp, c=colors[j], label=regions[j], lw=3)

      else:
        T_temp = T[exp][:,j,:]
        T_mean = np.mean(T_temp, axis=0)
        T_std = np.std(T_temp, axis=0)

        ax[i].plot(t, T_mean, c=colors[j], label='Ensemble Mean', linewidth=3)
        ax[i].fill_between(t, T_mean - T_std, T_mean + T_std, color=colors[j], alpha=0.5, label='Ensemble Spread (±1 std)')
        ax[i].fill_between(t, T_mean - 2*T_std, T_mean + 2*T_std, color=colors[j], alpha=0.2, label='Ensemble Spread (±2 std)')

    ax[i].set_title(f'{soln_type} - {experiments[i]}')
    ax[i].set_xlabel('Year')
    if not ensemble:
      ax[i].legend()

  ax[0].set_ylabel(r'$\Delta T$ [$^\circ$C]')

  return

def plot_box_model2(T, t, experiments, regions, colors, soln_type, coupled=False, ensemble=False, T2=None):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(2, 2, figsize=(12,12), sharex=True, sharey=True, constrained_layout=True)

  for i, exp in enumerate(experiments):
    if i == 0:
      r, c = 0, 0
    elif i == 1:
      r, c = 0, 1
    elif i == 2:
      r, c = 1, 0
    else:
      r, c = 1, 1

    for j, reg in enumerate(regions):
      if ensemble is False:
        if coupled:
          T_temp = np.array(T[exp])[j,:]
          T_temp_2 = np.array(T2[exp])[j,:]
        else:
          T_temp = T[exp][reg]
        n=15
        ax[r,c].plot(t, T_temp, c=brewer2_light(j), label=regions[j], lw=4)
        ax[r,c].plot(t[::n], T_temp_2[::n], 'o', c=brewer2_light(j), lw=4, markerfacecolor='white')

      else:
        T_temp = T[exp][:,j,:]
        T_mean = np.mean(T_temp, axis=0)
        T_std = np.std(T_temp, axis=0)

        ax[i].plot(t, T_mean, c=colors[j], label='Ensemble Mean', linewidth=3)
        ax[i].fill_between(t, T_mean - T_std, T_mean + T_std, color=colors[j], alpha=0.5, label='Ensemble Spread (±1 std)')
        ax[i].fill_between(t, T_mean - 2*T_std, T_mean + 2*T_std, color=colors[j], alpha=0.2, label='Ensemble Spread (±2 std)')

    ax[r,c].set_title(f'{experiments[i]}',fontsize=36)
    ax[r,c].tick_params(axis='both', which='major', labelsize=28)

  if not ensemble:
    custom_marker = Line2D(
    [0], [0],
    marker='o',
    color='black',             # Line color (ignored for the marker)
    markerfacecolor='white',   # White marker face
    markersize=10,
    linestyle='None'           # No line
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(custom_marker)
    labels.append('Emulator')
    ax[0,0].legend(handles,labels,fontsize=28,loc='upper left')

  #fig.suptitle('Experimental Overview',fontsize=32)

  ax[1,0].set_xlabel('Year',fontsize=36)
  ax[1,1].set_xlabel('Year',fontsize=36)
  ax[0,0].set_ylabel(r'$\Delta T$ [$^\circ$C]',fontsize=36)
  ax[1,0].set_ylabel(r'$\Delta T$ [$^\circ$C]',fontsize=36)

  plt.savefig('fig2b.pdf',dpi=500)

  return

def plot_G(G_raw, a, g, experiments, regions, colors):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(n_reg, n_exp, figsize=(5*n_exp,5*n_reg))

  for i in range(n_exp):
    exp = experiments[i]
    for j in range(n_reg):
      reg = regions[j]
      ax[i,j].plot(G_raw[exp][reg], c=colors[i], lw=2)
      ax[i,j].plot(g[exp][reg][0]*a[exp][reg][0], c=colors[i], ls='--', lw=2)

  return

def gen_F_ensemble(F, t, K, N_ensemble, experiments):
  F_ensemble = {}
  xi = np.random.normal(0, 1, (N_ensemble, len(t)))
  for exp in experiments:
    F_ensemble[exp] = K*xi + F[exp]

  return F_ensemble

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
          g_temp[n,:], a_temp[n, :] = calc_modes_1D(T_temp[n,:], t)
        g[exp][reg], a[exp][reg] = np.copy(g_temp), np.copy(a_temp)

      T[exp][reg] = np.copy(T_temp)

      if N_ensemble is None:
        g[exp][reg], a[exp][reg] = calc_modes_1D(T[exp][reg], t)

  return T, g, a

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
          g_temp[n,:], a_temp[n, :] = calc_modes_2D(T_temp[n,:,:])
        g[exp], a[exp] = np.copy(g_temp), np.copy(a_temp)

      T[exp] = np.copy(T_temp)

      if N_ensemble is None:
        g[exp], a[exp] = calc_modes_2D(T[exp])

  return T, g, a

def calc_L2(T_analytic, T_est, experiments, regions, mode, coupled=False):
  L2 = np.zeros(len(T_analytic) + 1)
  print(f'Error from {mode} estimation.')

  for i, exp in enumerate(experiments):
    L2[i] = 0

    if coupled:
      L2[i] = linalg.norm(T_analytic[exp] - T_est[exp])

    else:
      for reg in regions:
        L2[i] += linalg.norm(T_analytic[exp][reg] - T_est[exp][reg])

      L2[i] = L2[i]/(i + 1)

    print(f'\tL2 Error, {exp}: {np.round(L2[i], 5)}')

  L2[-1] = np.mean(L2[:-1])
  print(f'Avg. L2 Error: {np.round(L2[-1], 5)}\n')

  return L2

def calc_L2_operator(op_analytic, op_est, experiments, regions):
  L2_operator = {}

  for i, exp in enumerate(experiments):
    L2_operator[exp] = {}
    for j, reg in enumerate(regions):
      L2_operator[exp][j] = linalg.norm(op_analytic[exp][j] - op_est[exp][j])

  return L2_operator

def calc_G_deconv_1D(T, F, dt, modal=False):
  N_t = len(F)

  offsets = [i for i in range(0, -N_t, -1)]
  input_matrix = diags(F, offsets=offsets, shape=(N_t,N_t), format='csr')

  if modal:
    G = spsolve_triangular(input_matrix, T[0], lower=True)
  else:
    G = spsolve_triangular(input_matrix, T, lower=True)

  return G/dt

def opt_h_lam_1D(params, T, F, t, dt, modal=False):
    h, lam = params[0], params[1]
    G_opt = h*np.exp(-lam*t)

    if all(j == 0 for j in F):
      model = G_opt
    else:
      model = convolve(G_opt*dt, F)[:len(t)]

    model = np.roll(model,1)
    model[0] = 0

    return linalg.norm(T - model)

def apply_response_1D(params, t):
    h, lam = params[0], params[1]
    G = h*np.exp(-lam*t)
    G = np.roll(G,1)
    G[0] = 0
    return G

def calc_L2_operator_ensemble(op_analytic, op_ensemble, experiments, regions, N_ensemble, N_trials):
  L2_operator_ensemble = {}
  for exp in experiments:
    L2_operator_ensemble[exp] = {}
    for reg in regions:
      L2_operator_ensemble[exp][reg] = []

  # Loop over different number of ensemble members to sample
  for n in range(N_ensemble):

    L2_vals = {}
    for i, exp in enumerate(experiments):
      L2_vals[exp] = [[] * len(regions)]

    # Perform n_trials random samplings for each n
    for _ in range(N_trials):

      # Randomly sample n ensemble members without replacement
      sampled_indices = np.random.choice(N_ensemble, size=n+1, replace=False)
      op_sample_mean = {}

      for i, exp in enumerate(experiments):
        op_sample_mean[exp] = {}
        for j, reg in enumerate(regions):
          op_sample_mean[exp][j] = np.mean(op_ensemble[exp][j][sampled_indices, :], axis=0)

      L2_operator_sample = calc_L2_operator(op_analytic, op_sample_mean, experiments, regions)

      for i, exp in enumerate(experiments):
        for j, reg in enumerate(regions):
          L2_vals[exp][j].append(L2_operator_sample[exp][j])

    for i, exp in enumerate(experiments):
      for j, reg in enumerate(regions):
        L2_operator_ensemble[exp][j].append(np.mean(L2_vals[exp][j]))

  return L2_operator_ensemble

def plot_L2_operator_ensemble(L2_operator_ensemble, N_trials, experiments, regions, colors, soln_type):

  n_exp = len(experiments)
  n_reg = len(regions)
  fig, ax = plt.subplots(1, n_exp, figsize=(5*n_exp,5), sharey=True)

  for i in range(n_exp):
    exp = experiments[i]
    for j in range(n_reg):
      reg = regions[j]
      ax[i].loglog(N_trials, L2_operator_ensemble[exp][reg], c=colors[j], label=regions[j], lw=2)

    ax[i].set_title(f'{soln_type} - {experiments[i]}')
    ax[i].set_xlabel('No. Ensemble Members')
    ax[i].legend()

  ax[0].set_ylabel(r'$L_2$ Error [$L_\mathrm{True}$ - $L_\mathrm{Calc}$]')
  plt.tight_layout()

  return

#################### 2D Helper Functions ####################

def calc_derivative(f, t):
  M, N = f.shape
  df_dt = np.zeros((M, N))
  t = np.squeeze(t)

  for i in range(1, N - 1):
    t_fit = t[i - 1: i + 2]
    for m in range(M):
      f_fit = f[m, i - 1: i + 2]
      coeffs = np.polyfit(t_fit, f_fit, 2)
      a, b, _ = coeffs
      df_dt[m, i] = 2*a*t[i] + b

  df_dt[:, 0] = (f[:, 1] - f[:, 0]) / (t[1] - t[0])
  df_dt[:, -1] = (f[:, -1] - f[:, -2]) / (t[-1] - t[-2])

  return df_dt

def calc_L_direct_2D(T, F, t, gamma=None, modal=False, g=None, plot_L = False):
  R = np.copy(T)
  dR_dt = calc_derivative(R, t)
  R_p = linalg.pinv(R)

  if not is_pseudo_inverse(R, R_p):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)

    # Regularization parameter
    lambda_reg = 1e-6

    # Compute regularized pseudo-inverse
    s_reg = s / (s**2 + lambda_reg)
    R_p = Vt.T @ np.diag(s_reg) @ U.T

  if gamma is not None:
    F_spatial = np.zeros((len(gamma), len(F)))
    for i in range(len(gamma)):
      F_spatial[i,:] = gamma[i]*F
    F = F_spatial

  if modal:
    N = np.subtract(dR_dt, F) @ R_p
    L = g @ N @ linalg.pinv(g)
  else:
    L = np.subtract(dR_dt, F) @ R_p

  if plot_L:
    plt.figure(figsize=(10, 6))
    plt.imshow(L)
    plt.colorbar()

  return L

def reconstruct_T_2D(F, T, L, T0, dt, gamma=None):
  T_G = np.zeros(np.shape(T))
  T_G[:, 0] = T0

  if gamma is not None:
    F_spatial = np.zeros((len(gamma), len(F)))
    for i in range(len(gamma)):
      F_spatial[i,:] = gamma[i]*F
    F = F_spatial

  for i in range(1, np.shape(T)[1]):
    if gamma is not None:
      T_G[:, i] = (dt*L + np.identity(np.shape(L)[0])) @ T_G[:, i-1] + dt*F[:,i-1]
    else:
      T_G[:, i] = (dt*L + np.identity(np.shape(L)[0])) @ T_G[:, i-1] + dt*F[i-1]

  return T_G

def estimate_T_2D(T_analytic, F_all, experiments, t, mode, op_raw, op_modal, g, T0, dt, gamma=None):
  T_est_raw, T_est_modal = {}, {}

  # Iterate over forcing profiles
  for exp in experiments:
    F_toeplitz = toeplitz(F_all[exp], np.zeros_like(F_all[exp]))

    if mode == 'L':
      T_est_raw[exp] = reconstruct_T_2D(F_all[exp], T_analytic[exp], op_raw[exp], T0, dt, gamma)
      #T_est_modal[exp] = reconstruct_T_2D(F_all[exp], T_analytic[exp], op_modal[exp], T0, dt)

    elif mode == 'G':
      if gamma is not None:
        T_est_raw[exp] = []
        for i in range(len(T_analytic[exp])):
          T_est_raw[exp].append((op_raw[exp][i].T @ gamma) @ F_toeplitz.T)
      else:
        T_est_raw[exp] = (op_raw[exp]) @ F_toeplitz.T

  return T_est_raw, T_est_modal

# I use this method, written up this paper; you can talk to me about it after the talk
# Focus on the importance of the research
## 1 minute intro + context
## 1 minute methods
## 4 minutes results - essentially 4 figures (or 2 if they're really good)
## 2 minutes conclusions

# Risk showing something that's not ready for prime time
# Best bet to split half and half, here's the thing we did (JAMES) and here's where we're going
# 1st minute overall logic of the research (e.g. shortcomings of previous approaches)
# highlight JAMES paper first and foremost; show off the accomplishment!
# present the current work as an extension to that

# Making climate stripes for all CMIP - resolved spatially outputs, now cracking spatial inputs

# If system is more damped than the models say, less variability in the historical
# training on different realizations/reanalysis/historical data - estimating chaos in the system?
# To what degree does the geometry of the system impact the result? Lat/lon grid weighting needs to be taken into account


def calc_modes_2D(T):
  g, s, Vh = linalg.svd(T, full_matrices=False)
  a = s.reshape(len(s),1)*Vh
  return g, a

def calc_G_deconv_2D(T, F, dt, gamma=None):
  F_toeplitz = sparse.csr_matrix(toeplitz(F, np.zeros_like(F)))

  if gamma is not None:
    gamma = np.reshape(gamma, (len(gamma), 1))
    gamma_p = np.linalg.pinv(gamma)
    T = np.reshape(T, (len(T), 1))
    G = spsolve_triangular(F_toeplitz, T @ gamma_p, lower=True)

  else:
    G = spsolve_triangular(F_toeplitz, T.T, lower=True)

  return G.T/dt

def opt_h_lam_2D(params, T, F, t, m, dt):
  h = params[:m]
  lam = params[m]

  G_opt = h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])

  if all(j == 0 for j in F):
    model = G_opt
  else:
    model = convolve(G_opt*dt, F.reshape(1,len(F)))[:, :len(F)]

  return linalg.norm(T - model)

###### new
def opt_v_lam_2D(params, T, F, t, m, dt, gamma):
  # Assume the number of spatial points is the length of the gamma vector
  k = gamma.shape[0]

  # Extract eigenvectors (flattened into a 1D vector) and eigenvalues from the parameter vector
  # The first m*n values correspond to the eigenvectors (flattened)
  v = params[:m * k].reshape(k, m)  # Reshape to a matrix of size n x m
  w = np.linalg.pinv(v)             # Compute the pseudo-inverse of v (v^-1)
  lam = params[m * k:m * k + m]    # The next m values are the eigenvalues

  F_toeplitz = toeplitz(F, np.zeros_like(F))

  G_opt = np.zeros((k, len(t)))

  for i, n in enumerate(t):
    # Create diagonal matrix of exp(lambda_i * t) for this particular time t
    exp_diag_trunc = np.diag(np.exp(lam * n))

    # Compute exp(L * t) using eigenvalue expansion: v * exp(diag(lambda) * t) * v^-1
    exp_Lt_trunc = v @ exp_diag_trunc @ w

    # Compute G(x, t) by multiplying exp(L * t) with gamma
    G_opt[:, i] = np.dot(exp_Lt_trunc, gamma)

  model = (G_opt * dt) @ F_toeplitz.T

  # Account for weight in different boxes
  if True:
    weights = np.array([1,2,4])
    weighted_error = (T - model) * weights[:, np.newaxis]
    return linalg.norm(weighted_error)

  return linalg.norm(T - model)

def apply_v_lam_2D(params, t, m, gamma, dt):

  # Assume the number of spatial points is the length of the gamma vector
  k = gamma.shape[0]

  # Extract eigenvectors (flattened into a 1D vector) and eigenvalues from the parameter vector
  # The first m*n values correspond to the eigenvectors (flattened)
  v = params[:m * k].reshape(k, m)  # Reshape to a matrix of size n x m
  w = np.linalg.pinv(v)             # Compute the pseudo-inverse of v (v^-1)
  lam = params[m * k:m * k + m]    # The next m values are the eigenvalues

  G_opt = np.zeros((k, len(t)))

  for i, n in enumerate(t):
    # Create diagonal matrix of exp(lambda_i * t) for this particular time t
    exp_diag_trunc = np.diag(np.exp(lam * n))

    # Compute exp(L * t) using eigenvalue expansion: v * exp(diag(lambda) * t) * v^-1
    exp_Lt_trunc = v @ exp_diag_trunc @ w

    # Compute G(x, t) by multiplying exp(L * t) with gamma
    G_opt[:, i] = np.dot(exp_Lt_trunc, gamma)

  return G_opt*dt

###############

def apply_response_2D(params, t, m):
  h = params[:m]
  lam = params[m]
  return h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])

def opt_h_lam_2D_2(params, T, F, t, m, dt):
  L = np.reshape(params, (m,m))
  lam, h = np.linalg.eig(L)
  h_inv = np.linalg.inv(h)

  x0 = np.array([1, 0, 0])  # Initial condition
  G_opt = np.zeros((m, len(t)))
  for i, t_i in enumerate(t):
    # Compute e^(Lambda * t_i) - elementwise exponentiation of diagonal elements
    exp_Lambda_t = np.diag(np.exp(lam * t_i))

    # Compute P * e^(Lambda * t_i) * P^{-1}
    G_opt[:, i] = (h @ exp_Lambda_t @ h_inv) @ x0

  model = convolve(G_opt*dt, F.reshape(1,len(F)))[:, :len(F)]

  return linalg.norm(T - model)

def apply_response_2D_2(params, t, m):
  L = np.reshape(params, (m,m))
  lam, h = np.linalg.eig(L)
  h_inv = np.linalg.inv(h)
  x0 = np.array([1, 0, 0])  # Initial condition
  G_opt = np.zeros((m, len(t)))
  for i, t_i in enumerate(t):
    # Compute e^(Lambda * t_i) - elementwise exponentiation of diagonal elements
    exp_Lambda_t = np.diag(np.exp(lam * t_i))

    # Compute P * e^(Lambda * t_i) * P^{-1}
    G_opt[:, i] = (h @ exp_Lambda_t @ h_inv) @ x0

  return G_opt


def cumulative_h_lam_2D(params, target, forcing, t, m, dt, k):
  h = params[:k * m].reshape((k, m))  # First k*m elements are the h vectors
  lam = params[k * m:]  # Last k elements are the L scalars

  # Initialize the model with zeros
  model = np.zeros_like(target)

  # Add the contributions from each iteration
  for i in range(k):
    G_opt = h[i][:, np.newaxis]*np.exp(-lam[i]*t)*linalg.pinv(h[i][np.newaxis, :])
    model += convolve(G_opt*dt, forcing.reshape(1,len(forcing)))[:, :len(forcing)]

  return linalg.norm(model - target)

def iterative_minimization(target, forcing, t, m, dt, k):
  # Initialize the guess for h and L for the first iteration
  initial_h = np.ones(m)  # Initial guess for h_1
  initial_lam = 1        # Initial guess for L_1
  initial_guess = np.concatenate([initial_h, [initial_lam]])

  # Bounds for the variables: No bounds on h, and (0, 1) for L
  bounds = [(None, None)] * m + [(1e-10, 0.9999999999)]

  # Arrays to store the final results
  all_h = []
  all_lam = []

  for i in range(k):
      # Run the minimization for the current iteration
      result = minimize(cumulative_h_lam_2D, initial_guess, args=(target, forcing, t, m, dt, i + 1), bounds=bounds)

      # Extract optimized h and L
      optimized_h = result.x[:m * (i + 1)].reshape((i + 1, m))
      optimized_lam = result.x[m * (i + 1):]

      # Store the results
      all_h.append(optimized_h[-1])
      all_lam.append(optimized_lam[-1])

      # Prepare the initial guess for the next iteration
      initial_guess = np.concatenate([result.x, initial_h, [initial_lam]])
      bounds += [(None, None)] * m + [(1e-10, 0.9999999999)]

  return all_h, all_lam

def apply_response_cumulative_2D(h, lam, target, t, k):
  G_opt = np.zeros_like(target)

  for i in range(k):
    G_opt += h[i][:, np.newaxis]*np.exp(-lam[i]*t)*linalg.pinv(h[i][np.newaxis, :])

  return G_opt

def plot_2D(T, t_mesh, x_mesh, experiments, soln_type, T_2 = None):

  if T_2 is not None:
    import copy
    T_sub = {}
    for exp in experiments:
      T_sub[exp] = T[exp] - T_2[exp]
    T = copy.copy(T_sub)

  n_exp = len(experiments)
  fig, ax = plt.subplots(1, n_exp, figsize=(5*n_exp,5), sharey=True, sharex=True)

  # Compute the global min and max across all experiments
  global_min = np.min([np.min(T[exp]) for exp in experiments])
  global_max = np.max([np.max(T[exp]) for exp in experiments])
  colorbars = []

  for i, exp in enumerate(experiments):
    if T_2 is not None:
      cf = ax[i].contourf(t_mesh, x_mesh, T[exp], vmin=global_min, vmax=global_max, cmap='RdBu')
    else:
      cf = ax[i].contourf(t_mesh, x_mesh, T[exp], vmin=global_min, vmax=global_max)
    colorbars.append(cf)
    ax[i].set_title(f'{soln_type} - {experiments[i]}')
    ax[i].set_xlabel('Time (year)')

  ax[0].set_ylabel('Location (m)')

  cbar = fig.colorbar(colorbars[2], ax=ax, orientation='vertical', fraction=0.025, pad=0.04)
  cbar.set_label('Temperature (or other variable)')

  #plt.tight_layout()

  return

def plot_2D2(T, t_mesh, x_mesh, experiments, soln_type, T_2 = None):

  if T_2 is not None:
    import copy
    T_sub = {}
    for exp in experiments:
      T_sub[exp] = T[exp] - T_2[exp]
    T = copy.copy(T_sub)

  n_exp = len(experiments)
  fig, ax = plt.subplots(2, 2, figsize=(12,12), sharey=True, sharex=True, constrained_layout=True)

  # Compute the global min and max across all experiments
  global_min = np.min([np.min(T[exp]) for exp in experiments])
  global_max = np.max([np.max(T[exp]) for exp in experiments])
  colorbars = []

  for i, exp in enumerate(experiments):
    if i == 0:
      r, c = 0, 0
    elif i == 1:
      r, c = 0, 1
    elif i == 2:
      r, c = 1, 0
    else:
      r, c = 1, 1
    if T_2 is not None:
      cf = ax[r,c].contourf(x_mesh, t_mesh, T[exp], vmin=global_min, vmax=global_max, cmap='RdBu')
    else:
      cf = ax[r,c].contourf(x_mesh, t_mesh, T[exp], vmin=global_min, vmax=global_max)

    colorbars.append(cf)
    ax[r,c].set_title(f'{experiments[i]}',fontsize=36)
    ax[r,c].tick_params(axis='both', which='major', labelsize=24)
    ax[r,c].get_xaxis().set_ticks([])
    ax[r,c].get_yaxis().set_ticks([])
  
  ax[0,0].set_ylabel(r'Time $(t)$',fontsize=36)
  ax[1,0].set_ylabel(r'Time $(t)$',fontsize=36)

  ax[1,0].set_xlabel(r'Position ($\mathbf{x}$)',fontsize=36)
  ax[1,1].set_xlabel(r'Position ($\mathbf{x}$)',fontsize=36)

  cbar = fig.colorbar(colorbars[2], ax=ax, orientation='vertical', fraction=0.075, pad=0.02)
  cbar.ax.tick_params(labelsize=28) 

  cbar.set_label(r'$\Delta T$ [$^\circ$C]',fontsize=36)

  #plt.savefig('fig3.pdf',dpi=500)
  return

  ################### Other

def finite_difference_derivative(f, t, order=1):
  """
  Approximate the derivative of a vector f with respect to t using finite differences.

  Parameters:
  f (array-like): The input vector of function values.
  t (array-like): The input vector of time or spatial values (must be evenly spaced).
  order (int): The order of the finite difference approximation (1, 2, or 3).

  Returns:
  df_dt (numpy array): The approximate derivative of f with respect to t.
  """
  if order not in [1, 2, 3, 4]:
    raise ValueError("Order must be 1, 2, 3, or 4.")

  N = len(f)
  df_dt = np.zeros(N)
  dt = t[1] - t[0]  # Assume evenly spaced values

  if order == 1:
    # First-order backward difference for boundary
    df_dt[-1] = (f[-1] - f[-2]) / dt

    for i in range(0, N - 1):
      df_dt[i] = (f[i + 1] - f[i]) / dt

  elif order == 2:
    # Second-order forward difference for the first point
    df_dt[0] = (-3*f[0] + 4*f[1] - f[2]) / (2 * dt)

    # Second-order backward difference for the last point
    df_dt[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2 * dt)

    # Central difference for true interior points
    for i in range(1, N - 1):
      df_dt[i] = (f[i + 1] - f[i - 1]) / (2 * dt)

  elif order == 3:
    # Third-order backward difference for the last three points
    df_dt[-3:] = (11*f[-3:] - 18*f[-4:-1] + 9*f[-5:-2] - 2*f[-6:-3]) / (6 * dt)

    # Central difference for true interior points
    for i in range(0, N - 3):
      df_dt[i] = (-11*f[i] + 18*f[i+1] - 9*f[i+2] + 2*f[i+3]) / (6 * dt)

  elif order == 4:
    # Fourth-order forward difference for the first four points
    df_dt[0:4] = (-25*f[0:4] + 48*f[1:5] - 36*f[2:6] + 16*f[3:7] - 3*f[4:8]) / (12 * dt)

    # Fourth-order backward difference for the last four points
    df_dt[-4:] = (25*f[-4:] - 48*f[-5:-1] + 36*f[-6:-2] - 16*f[-7:-3] + 3*f[-8:-4]) / (12 * dt)

    # Central difference for true interior points
    for i in range(3, N - 4):
      df_dt[i] = (-f[i + 2] + 8*f[i + 1] - 8*f[i - 1] + f[i - 2]) / (12 * dt)

  return df_dt

def scatter_plot(data):
    """
    Creates a scatter plot for error values at different locations.
    
    Parameters:
    data (dict): A dictionary where keys are locations (str) and values are lists
                 of 4 error values (float) corresponding to different variables.
    """
    locations = list(data.keys())
    x_vals = []
    y_vals = []
    labels = []
    colors = ['red', 'blue', 'green', 'purple']
    
    # Organize data for plotting
    for i, location in enumerate(locations):
      for j, error_value in enumerate(data[location]):
        x_vals.append(i + 1)  # x-axis positions (1, 2, 3, ..., corresponding to locations)
        y_vals.append(error_value)
        #labels.append(f"{location}_Var{j+1}")  # Label for each point

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
      plt.scatter(x, y, color=colors[i % len(colors)], alpha=0.7)

    # Add labels and title
    plt.xlabel("Location")
    plt.ylabel("Error Value")
    plt.title("Scatter Plot of Error Values at Different Locations")
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    

    plt.show()



def is_pseudo_inverse(A, A_pseudo_inv):
    """
    Check if A_pseudo_inv is the pseudo-inverse of A by verifying the Moore-Penrose conditions.

    Parameters:
    A : numpy.ndarray
        The original matrix.
    A_pseudo_inv : numpy.ndarray
        The candidate pseudo-inverse of A.

    Returns:
    bool
        True if A_pseudo_inv is a valid pseudo-inverse of A, False otherwise.
    """
    # Condition 1: AA^+A = A
    condition1 = np.allclose(A @ A_pseudo_inv @ A, A)

    # Condition 2: A^+AA^+ = A^+
    condition2 = np.allclose(A_pseudo_inv @ A @ A_pseudo_inv, A_pseudo_inv)

    # Condition 3: (AA^+)^T = AA^+
    condition3 = np.allclose((A @ A_pseudo_inv).T, A @ A_pseudo_inv)

    # Condition 4: (A^+A)^T = A^+A
    condition4 = np.allclose((A_pseudo_inv @ A).T, A_pseudo_inv @ A)

    return condition1 and condition2 and condition3 and condition4



######################## Plot Colormap ###########################
# Credit: https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=8
brewer2_light_rgb = np.divide([(102, 194, 165),
                               (252, 141,  98),
                               (141, 160, 203),
                               (231, 138, 195),
                               (166, 216,  84),
                               (255, 217,  47),
                               (229, 196, 148),
                               (179, 179, 179),
                               (202, 178, 214)],255)
brewer2_light = mcolors.ListedColormap(brewer2_light_rgb)
