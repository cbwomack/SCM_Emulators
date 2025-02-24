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


import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


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

# Sinusoidal forcing
def F_sin(t, F0):
  return F0*np.sin(t)

# Delta forcing
def F_del(t, F0):
  return F0*unit_impulse(len(t))

# Overshoot forcing
def F_over(t, a, b, c):
  return a*np.exp(-np.square(t - b)/(2*c**2))

##### Solutions given the above forcing profiles
# General ODE 1D: C*dT(t)/dt = -lam*T(t) + F(t)
# or
# General ODE 2D: dT(x,t)/dt = -k(x)*T(x,t) + F(t)
# or
# General ODE 3D: dT(x,y,t)/dt = -k(x,y)*T(x,y,t) + F(t)

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

# F(t) = a*exp(t/ts)
def T_ramp(t, t_ramp, ts, a, T0, lam, C, dim=1):
  if dim == 1:
    return np.concatenate((np.zeros(len(t_ramp)), T_exp(t, ts, a, T0, lam, C)))
  else:
    raise ValueError("dim must be 1.")

# F(t) = F*delta(t)
def T_del(t, lam, F0, C, dt, dim=1):
  if dim == 1:
    T_d = np.roll(np.exp(t*(lam/C))*(np.heaviside(t, F0))/(C/dt),1)
    #T_d = np.exp(t*(lam/C))*(np.heaviside(t, F0))/(C/dt)
    T_d[0] = 0
    return T_d
  else:
    raise ValueError("dim must be 1")

"""
# F(t) = F
def T_const(t, T0, k, F0, dim=1, x=None, y=None):
  if dim == 1:
    return F0/k + (-F0 + T0*k)*np.exp(-k*t)/k
  elif dim == 2:
    if x is not None:
      return F0/(k*x) + (-F0 + T0*k*x)*np.exp(-k*x*t)/(k*x)
    else:
      raise ValueError("For dim=2, x must be provided.")
  elif dim == 3:
    if x is not None and y is not None:
      xy = np.outer(x, y).reshape(-1, 1)
      return F0/(k*xy) + (-F0 + T0*k*xy)*np.exp(-k*xy*t)/(k*xy)
    else:
      raise ValueError("For dim=3, both x and y must be provided.")
  else:
    raise ValueError("dim must be 1, 2, or 3.")
"""

# F(t) = F*sin(t)
def T_sin(t, T0, k, F0, dim=1, x=None, y=None):
  if dim == 1:
    return np.exp(-k*t)*(T0 + np.exp(k*t)*F0*(-np.cos(t) + k*np.sin(t))/(k**2 + 1))
  elif dim == 2:
    if x is not None:
      return np.exp(-k*x*t)*(T0 + np.exp(k*x*t)*F0*(-np.cos(t) + k*x*np.sin(t))/(np.power(k*x,2) + 1))
    else:
      raise ValueError("For dim=2, x must be provided.")
  elif dim == 3:
    if x is not None and y is not None:
      xy = np.outer(x, y).reshape(-1, 1)
      return np.exp(-k*xy*t)*(T0 + np.exp(k*xy*t)*F0*(-np.cos(t) + k*xy*np.sin(t))/(np.power(k*xy,2) + 1))
    else:
      raise ValueError("For dim=3, both x and y must be provided.")
  else:
    raise ValueError("dim must be 1, 2, or 3.")

"""
# F(t) = F*delta(t)
def T_del(t, k, F0, dim=1, x=None, y=None):
  if dim == 1:
    return np.exp(-k*t)*(np.heaviside(t, F0))
  elif dim == 2:
    if x is not None:
      return np.exp(-k*x*t)*(np.heaviside(t, F0))
    else:
      raise ValueError("For dim=2, x must be provided.")
  elif dim == 3:
    if x is not None and y is not None:
      xy = np.outer(x, y).reshape(-1, 1)
      return np.exp(-k*xy*t)*(np.heaviside(t, F0))
    else:
      raise ValueError("For dim=3, both x and y must be provided.")
  else:
    raise ValueError("dim must be 1, 2, or 3.")
"""

def calc_temp_response_1D(exp, t, T0, F_2xCO2, F_4xCO2, ts, a_exp, a_over, b_over, c_over, lam, C):
  if exp == '2xCO2': return T_const(t, T0, lam, F_2xCO2, C)
  elif exp == '4xCO2': return T_const(t, T0, lam, F_4xCO2, C)
  elif exp == 'RCP70': return T_exp(t, ts, a_exp, T0, lam, C)
  elif exp == 'Overshoot': return T_over(t, T0, a_over, b_over, c_over, lam, C)
  else:
    raise ValueError(f"exp: {exp} not found")

##### Solver Methodologies

# Reconstruct temperature profile given a linear operator
def reconstruct_T(F, t, L, T0, dt):
  N_t = len(t)
  T_G = np.zeros(N_t)
  T_G[0] = T0

  for i in range(1, N_t):
    T_G[i] = (dt*L[0][0] + 1)*T_G[i-1] + dt*F[i-1]

  return T_G

def reconstruct_T_C(F, C, t, L, T0, dt):
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

def calc_L_direct_1D(T, F, t, modal=False, g=None):

  R = np.copy(T).reshape(1,len(t))
  dR_dt = calc_derivative(R, t)
  R_p = linalg.pinv(R)

  if modal:
    if np.shape(g) != (1,1):
      g = g.reshape(1,1)
    N = np.dot(np.subtract(dR_dt, F), R_p)
    L = g[0]*N*linalg.pinv(g)[0]
  else:
    L = np.dot(np.subtract(dR_dt, F), R_p)

  return L

def calc_L_direct_1D_C(T, F, t, C, modal=False, g=None):

  R = np.copy(T).reshape(1,len(t))
  dR_dt = calc_derivative(R, t)
  R_p = linalg.pinv(R)

  if modal:
    if np.shape(g) != (1,1):
      g = g.reshape(1,1)
    N = np.dot(np.subtract(dR_dt, np.divide(F,C)), R_p)
    L = g[0]*N*linalg.pinv(g)[0]
  else:
    L = np.dot(np.subtract(dR_dt, np.divide(F,C)), R_p)

  return L

def plot_response_1D_region(T_vec, F_vec, labels, t, L_raw, L_modal, T0, dt, L_ind=0):
  T_est_raw, T_est_modal = np.zeros(np.shape(T_vec)), np.zeros(np.shape(T_vec))

  colors = ['g','r','b']
  fig, ax = plt.subplots(3, len(T_vec), figsize=(15,5*(len(T_vec))), sharey=True, sharex=True)

  # Iterate over forcing profiles
  for i in [0,1]:#range(len(T_vec)):
    # Iterate over regions
    for j in [0,1,2]:#range(len(T_vec[i])):
      T_est_raw[i][j] = reconstruct_T(F_vec[i], t, L_raw[j], T0, dt)
      T_est_modal[i][j] = reconstruct_T(F_vec[i], t, -L_modal[j], T0, dt)

      ax[0][i].plot(T_vec[i][j], c=colors[j], lw=2, label=labels[j])
      ax[0][i].legend()
      ax[1][i].plot(T_est_raw[i][j], c=colors[j], ls='--', lw=2, label=labels[j])
      ax[1][i].legend()
      ax[2][i].plot(T_est_modal[i][j], c=colors[j], ls='-.', lw=2, label=labels[j])
      ax[2][i].legend()

  ax[0][0].set_title(r'Analytic - \textit{2xCO2}')
  ax[0][1].set_title(r'Analytic - \textit{4xCO2}')
  ax[0][2].set_title(r'Analytic - Exponential (7 Wm$^{-2}$)')

  ax[1][0].set_title(r'Raw $L$ - \textit{2xCO2}')
  ax[1][1].set_title(r'Raw $L$ - \textit{4xCO2}')
  ax[1][2].set_title(r'Raw $L$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_title(r'Modal $L$ - \textit{2xCO2}')
  ax[2][1].set_title(r'Modal $L$ - \textit{4xCO2}')
  ax[2][2].set_title(r'Modal $L$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_xlabel('Year')
  ax[2][1].set_xlabel('Year')
  ax[2][2].set_xlabel('Year')

  ax[0][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[1][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[2][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')

  #L2_raw = calc_L2(T_vec, T_est_raw, labels, 'raw')
  #L2_modal = calc_L2(T_vec, T_est_modal, labels, 'modal')
  plt.tight_layout()

  return #L2_raw, L2_modal

def plot_response_1D_region_C(T_vec, F_vec, labels, t, C, L_raw, L_modal, T0, dt, L_ind=0):
  T_est_raw, T_est_modal = np.zeros(np.shape(T_vec)), np.zeros(np.shape(T_vec))

  colors = ['g','r','b']
  fig, ax = plt.subplots(3, len(T_vec), figsize=(15,5*(len(T_vec))), sharey=True, sharex=True)

  # Iterate over forcing profiles
  for i in [0,1,2]:#range(len(T_vec)):
    # Iterate over regions
    for j in [0,1,2]:#range(len(T_vec[i])):
      T_est_raw[i][j] = reconstruct_T_C(F_vec[i], C[j], t, L_raw[j], T0, dt)
      T_est_modal[i][j] = reconstruct_T_C(F_vec[i], C[j], t, L_modal[j], T0, dt)

      ax[0][i].plot(T_vec[i][j], c=colors[j], lw=2, label=labels[j])
      ax[0][i].legend()
      ax[1][i].plot(T_est_raw[i][j], c=colors[j], ls='--', lw=2, label=labels[j])
      ax[1][i].legend()
      ax[2][i].plot(T_est_modal[i][j], c=colors[j], ls='-.', lw=2, label=labels[j])
      ax[2][i].legend()

  ax[0][0].set_title(r'Analytic - \textit{2xCO2}')
  ax[0][1].set_title(r'Analytic - \textit{4xCO2}')
  ax[0][2].set_title(r'Analytic - Exponential (7 Wm$^{-2}$)')

  ax[1][0].set_title(r'Raw $L$ - \textit{2xCO2}')
  ax[1][1].set_title(r'Raw $L$ - \textit{4xCO2}')
  ax[1][2].set_title(r'Raw $L$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_title(r'Modal $L$ - \textit{2xCO2}')
  ax[2][1].set_title(r'Modal $L$ - \textit{4xCO2}')
  ax[2][2].set_title(r'Modal $L$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_xlabel('Year')
  ax[2][1].set_xlabel('Year')
  ax[2][2].set_xlabel('Year')

  ax[0][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[1][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[2][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')

  plt.tight_layout()

  return

def plot_response_1D_conv_region(T_vec, F_vec, labels, t, G_raw, G_modal, g, dt, check=True):
  T_est_raw, T_est_modal = np.zeros(np.shape(T_vec)), np.zeros(np.shape(T_vec))

  colors = ['g','r','b']
  fig, ax = plt.subplots(3, len(T_vec), figsize=(15,5*(len(T_vec))), sharey=True, sharex=True)

  # Iterate over forcing profiles
  for i in range(len(T_vec)):
    # Iterate over regions
    for j in [0,1,2]:#range(len(T_vec[i])):
      T_est_raw[i][j] = convolve(G_raw[j]*dt, F_vec[i])[:len(t)]
      if check:
        T_est_modal[i][j] = g[j][0]*convolve(G_modal[j][0]*dt, F_vec[i])[:len(t)]
      else:
        T_est_modal[i][j] = g[j]*convolve(G_modal[j]*dt, F_vec[i])[:len(t)]

      ax[0][i].plot(T_vec[i][j], c=colors[j], lw=2, label=labels[j])
      ax[0][i].legend()
      ax[1][i].plot(T_est_raw[i][j], c=colors[j], ls='--', lw=2, label=labels[j])
      ax[1][i].legend()
      ax[2][i].plot(T_est_modal[i][j], c=colors[j], ls='-.', lw=2, label=labels[j])
      ax[2][i].legend()

  ax[0][0].set_title(r'Analytic - \textit{2xCO2}')
  ax[0][1].set_title(r'Analytic - \textit{4xCO2}')
  ax[0][2].set_title(r'Analytic - Exponential (7 Wm$^{-2}$)')

  ax[1][0].set_title(r'Raw $G$ - \textit{2xCO2}')
  ax[1][1].set_title(r'Raw $G$ - \textit{4xCO2}')
  ax[1][2].set_title(r'Raw $G$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_title(r'Modal $G$ - \textit{2xCO2}')
  ax[2][1].set_title(r'Modal $G$ - \textit{4xCO2}')
  ax[2][2].set_title(r'Modal $G$ - Exponential (7 Wm$^{-2}$)')

  ax[2][0].set_xlabel('Year')
  ax[2][1].set_xlabel('Year')
  ax[2][2].set_xlabel('Year')

  ax[0][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[1][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')
  ax[2][0].set_ylabel(r'$\Delta T$ [$^\circ$C]')

  #L2_raw = calc_L2(T_vec, T_est_raw, labels, 'raw')
  #L2_modal = calc_L2(T_vec, T_est_modal, labels, 'modal')
  plt.tight_layout()

  return #L2_raw, L2_modal

def plot_response_1D(T_vec, F_vec, labels, t, L_raw, L_modal, T0, dt):
  T_est_raw, T_est_modal = np.zeros((len(T_vec), len(t))), np.zeros((len(T_vec), len(t)))

  colors = ['b','r','m','g']
  fig, ax = plt.subplots()
  for i in range(len(T_vec)):
    T_est_raw[i] = reconstruct_T(F_vec[i], t, L_raw, T0, dt)
    T_est_modal[i] = reconstruct_T(F_vec[i], t, L_modal, T0, dt)

    ax.plot(t, T_vec[i], c=colors[i], lw=2, label=labels[i])
    ax.plot(t, T_est_raw[i], c=colors[i], ls='--', lw=2, label=f'Recon. {labels[i]}: raw')
    ax.plot(t, T_est_modal[i], c=colors[i], ls='-.', lw=2, label=f'Recon. {labels[i]}: modal')

  ax.legend()

  L2_raw = calc_L2(T_vec, T_est_raw, labels, 'raw')
  L2_modal = calc_L2(T_vec, T_est_modal, labels, 'modal')

  return T_est_raw#L2_raw, L2_modal

def plot_response_conv_1D(T_vec, F_vec, labels, t, G_raw, G_modal, g, dt):
  T_est_raw, T_est_modal = np.zeros((len(T_vec) - 1, len(t))), np.zeros((len(T_vec) - 1, len(t)))

  colors = ['b','r','m','g']
  fig, ax = plt.subplots()
  for i in range(1, len(T_vec)):
    T_est_raw[i - 1] = convolve(G_raw*dt, F_vec[i])[:len(t)]
    T_est_modal[i - 1] = g*convolve(G_modal*dt, F_vec[i])[:len(t)]

    ax.plot(t, T_vec[i] - T_vec[0], c=colors[i], lw=2, label=labels[i])
    ax.plot(t, T_est_raw[i - 1], c=colors[i], ls='--', lw=2, label=f'Recon. {labels[i]}: raw')
    ax.plot(t, T_est_modal[i - 1], c=colors[i], ls='-.', lw=2, label=f'Recon. {labels[i]}: modal')

  ax.legend()

  L2_raw = calc_L2(T_vec[1:] - T_vec[0], T_est_raw, labels[1:], 'raw')
  L2_modal = calc_L2(T_vec[1:] - T_vec[0], T_est_modal, labels[1:], 'modal')

  return L2_raw, L2_modal

def calc_L2(T_vec, T_est, labels, mode):
  L2 = np.zeros(len(T_vec) + 1)
  print(f'Error from {mode} estimation.')
  for i in range(len(T_vec)):
    L2[i] = linalg.norm(T_vec[i] - T_est[i])
    print(f'\tL2 Error, {labels[i]}: {np.round(L2[i], 5)}')

  L2[-1] = np.mean(L2[:-1])
  print(f'Avg. L2 Error: {np.round(L2[-1], 5)}\n')
  return L2

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

def calc_L_direct_2D(T, F, t, modal=False, g=None, plot_L = False):
  R = np.copy(T)
  dR_dt = calc_derivative(R, t)
  R_p = linalg.pinv(R)

  if modal:
    N = np.dot(np.subtract(dR_dt, F), R_p)
    L = np.dot(np.dot(g,N),linalg.pinv(g))
  else:
    L = np.dot(np.subtract(dR_dt, F), R_p)

  if plot_L:
    plt.figure(figsize=(10, 6))
    plt.imshow(L)
    plt.colorbar()

  return L

def reconstruct_T_2D(F, T, L, T0, dt):
  T_G = np.zeros(np.shape(T))
  T_G[:, 0] = T0
  for i in range(1, np.shape(T)[1]):
    T_G[:, i] = np.dot((dt*L + np.identity(np.shape(L)[0])), T_G[:, i-1]) + dt*F[i-1]

  return T_G

def calc_modes_2D(T):
  g, s, Vh = linalg.svd(T, full_matrices=False)
  a = s.reshape(len(s),1)*Vh
  return g, a

def calc_G_deconv_2D(T, F, dt):
  N_t = len(F)
  offsets = [i for i in range(0, -N_t, -1)]
  input_matrix = diags(F, offsets=offsets, shape=(N_t, N_t), format='csr')
  G = spsolve_triangular(input_matrix, T.T, lower=True)

  return G.T/dt

def plot_response_2D(T_vec, F_vec, labels, t_mesh, x_mesh, mode, op_raw, op_modal, T0, g, dt):
  N_temp, N_t, N_x = np.shape(T_vec)
  if mode == 'L':
    T_est_raw, T_est_modal = np.zeros((N_temp, N_t, N_x)), np.zeros((N_temp, N_t, N_x))
  elif mode == 'G':
    T_est_raw, T_est_modal = np.zeros((N_temp - 1, N_t, N_x)), np.zeros((N_temp - 1, N_t, N_x))

  fig, ax = plt.subplots(3, len(T_est_raw), figsize=(15,5*(len(T_est_raw))), sharex = True, sharey=True)
  if len(T_est_raw) == 1: ax = np.expand_dims(ax, axis=1)

  if mode == 'L':
    vmin = np.min(T_vec)
    vmax = np.max(T_vec)
  elif mode == 'G':
    vmin = np.min(T_vec[1:] - T_vec[0])
    vmax = np.max(T_vec[1:] - T_vec[0])

  for i in range(len(T_est_raw)):
    if mode == 'L':
      T_est_raw[i, :, :] = reconstruct_T_2D(F_vec[i], T_vec[i], op_raw, T0, dt)
      T_est_modal[i, :, :] = reconstruct_T_2D(F_vec[i], T_vec[i], op_modal, T0, dt)

      c1 = ax[0, i].contourf(t_mesh, x_mesh, T_vec[i], vmin=vmin, vmax=vmax)
      c2 = ax[1, i].contourf(t_mesh, x_mesh, T_est_raw[i], vmin=vmin, vmax=vmax)
      c3 = ax[2, i].contourf(t_mesh, x_mesh, T_est_modal[i], vmin=vmin, vmax=vmax)

      j = i

    elif mode == 'G':
      T_est_raw[i, :, :] = convolve(op_raw*dt, F_vec[i + 1].reshape(1,len(F_vec[i + 1])))[:, :len(F_vec[i + 1])]
      T_est_modal[i, :, :] = np.dot(g, convolve(op_modal*dt, F_vec[i + 1].reshape(1,len(F_vec[i + 1])))[:, :len(F_vec[i + 1])])

      c1 = ax[0, i].contourf(t_mesh, x_mesh, T_vec[i + 1] - T_vec[0], vmin=vmin, vmax=vmax)
      c2 = ax[1, i].contourf(t_mesh, x_mesh, T_est_raw[i], vmin=vmin, vmax=vmax)
      c3 = ax[2, i].contourf(t_mesh, x_mesh, T_est_modal[i], vmin=vmin, vmax=vmax)

      j = i + 1

    else:
      raise ValueError('Mode needs to be either L or G!')

    ax[0, i].set_title(rf'Analytic: {labels[j]}')
    ax[1, i].set_title(rf'Raw {mode}: {labels[j]}')
    ax[2, i].set_title(rf'Modal {mode}: {labels[j]}')

  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  sm = mpl.cm.ScalarMappable(cmap=c1.cmap, norm=norm)
  sm.set_array([])

  fig.colorbar(sm, ax=ax.ravel().tolist(), orientation='vertical')

  if mode == 'L':
    L2_raw = calc_L2(T_vec, T_est_raw, labels, 'raw')
    L2_modal = calc_L2(T_vec, T_est_modal, labels, 'modal')
  if mode == 'G':
    L2_raw = calc_L2(T_vec[1:] - T_vec[0], T_est_raw, labels[1:], 'raw')
    L2_modal = calc_L2(T_vec[1:] - T_vec[0], T_est_modal, labels[1:], 'modal')

  return L2_raw, L2_modal

def plot_response_conv_2D(T_vec, F_vec, labels, t, x, G_raw, G_modal, g, dt):
  shape = np.shape(T_vec)
  T_est_raw, T_est_modal = np.zeros((shape[0] - 1, shape[1], shape[2])), np.zeros((shape[0] - 1, shape[1], shape[2]))

  fig, ax = plt.subplots(3, len(T_vec), figsize=(15,5*(len(T_vec))), sharex = True, sharey=True)
  for i in range(1, len(T_vec)):
    T_est_raw[i - 1, :, :] = convolve(G_raw*dt, F_vec[i].reshape(1,len(F_vec[i])))[:, :len(F_vec[i])]
    T_est_modal[i - 1, :, :] = np.dot(g, convolve(G_modal*dt, F_vec[i].reshape(1,len(F_vec[i])))[:, :len(F_vec[i])])

    ax[0, i].contourf(t, x[:, 0], T_vec[i] - T_vec[0])
    ax[0, i].set_title(labels[i])
    ax[1, i].contourf(t, x[:, 0], T_est_raw[i - 1])
    ax[1, i].set_title(f'Recon. {labels[i]}: raw')
    ax[2, i].contourf(t, x[:, 0], T_est_modal[i - 1])
    ax[2, i].set_title(f'Recon. {labels[i]}: modal')

  L2_raw = calc_L2(T_vec[1:] - T_vec[0], T_est_raw, labels[1:], 'raw')
  L2_modal = calc_L2(T_vec[1:] - T_vec[0], T_est_modal, labels[1:], 'modal')

  return L2_raw, L2_modal

def opt_h_lam_2D(params, T, F, t, m, dt):
  h = params[:m]
  lam = params[m]

  G_opt = h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])

  if all(j == 0 for j in F):
    model = G_opt
  else:
    model = convolve(G_opt*dt, F.reshape(1,len(F)))[:, :len(F)]

  return linalg.norm(T - model)

def apply_response_2D(params, t, m):
  h = params[:m]
  lam = params[m]
  return h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])


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

#################### 3D Helper Functions ####################

def plot_response_3D(T_vec, F_vec, labels, t, x, y, L_raw, L_modal, T0, dt, plot_time=0, animate=False, num_frames=100, playback_speed=100):
  T_est_raw, T_est_modal = np.zeros(np.shape(T_vec)), np.zeros(np.shape(T_vec))

  Nx, Ny = len(x), len(y[0])

  # Set up the figure and axes
  fig, ax = plt.subplots(len(T_vec), 3, figsize=(15, 5 * len(T_vec)), sharex=True, sharey=True)
  plt.subplots_adjust(wspace=0.4, hspace=0.4)

  # Prepare X, Y meshgrid
  X, Y = np.meshgrid(x, y)

  # Create a placeholder for the colorbar
  cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

  for i in range(len(T_vec)):
    T_est_raw[i, :, :] = reconstruct_T_2D(F_vec[i], T_vec[i], L_raw, T0, dt)
    T_est_modal[i, :, :] = reconstruct_T_2D(F_vec[i], T_vec[i], L_modal, T0, dt)

  # Calculate the global min and max for consistent color mapping
  vmin = min(T_vec[0].min(), T_vec[1].min(), T_vec[2].min())
  vmax = max(T_vec[0].max(), T_vec[1].max(), T_vec[2].max())

  for i in range(len(T_vec)):
        # Reshape the vectors from (Nx*Ny, Nt) to (Nx, Ny, Nt)
    T_vec_reshaped = T_vec[i].reshape(Nx, Ny, -1)
    T_est_raw_reshaped = T_est_raw[i].reshape(Nx, Ny, -1)
    T_est_modal_reshaped = T_est_modal[i].reshape(Nx, Ny, -1)

    if not animate:
      # Select the time slice for contour plotting
      time_label = f"t = {t[plot_time]:.2f}"

      # Ensure x and y are properly shaped
      X, Y = np.meshgrid(x, y)

      # Contour plots
      cont0 = ax[i, 0].contourf(X, Y, T_vec_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i, 0].set_title(f"{labels[i]} - {time_label}")
      cont1 = ax[i, 1].contourf(X, Y, T_est_raw_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i, 1].set_title(f'Recon. {labels[i]}: raw - {time_label}')
      cont2 = ax[i, 2].contourf(X, Y, T_est_modal_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i, 2].set_title(f'Recon. {labels[i]}: modal - {time_label}')

      # Set labels
      for j in range(3):
        ax[i, j].set_xlabel('x')
        ax[i, j].set_ylabel('y')

  if not animate:
    fig.colorbar(cont0, cax=cbar_ax)
    plt.show()

  def anim_3D(k):
    for i in range(len(T_vec)):
      time_label = f"t = {t[k]:.2f}"

      # Clear the axes to redraw
      ax[i, 0].cla()
      ax[i, 1].cla()
      ax[i, 2].cla()

      # Contour plots
      cont0 = ax[i, 0].contourf(X, Y, T_vec_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i, 0].set_title(f"{labels[i]} - {time_label}")
      cont1 = ax[i, 1].contourf(X, Y, T_est_raw_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i, 1].set_title(f'Recon. {labels[i]}: raw - {time_label}')
      cont2 = ax[i, 2].contourf(X, Y, T_est_modal_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i, 2].set_title(f'Recon. {labels[i]}: modal - {time_label}')

      # Set labels
      for j in range(3):
        ax[i, j].set_xlabel('x')
        ax[i, j].set_ylabel('y')

    # Set the colorbar only once (for the first frame)
    if k == 0:
        fig.colorbar(cont0, cax=cbar_ax)

  if animate:
    anim = animation.FuncAnimation(
        fig,
        anim_3D,
        frames=num_frames,          # Number of frames to animate
        interval=playback_speed,    # Interval in milliseconds between frames
        repeat=False
    )
    plt.show()

    return anim

  else:
    L2_raw = calc_L2(T_vec, T_est_raw, labels, 'raw')
    L2_modal = calc_L2(T_vec, T_est_modal, labels, 'modal')

    return L2_raw, L2_modal

def plot_response_conv_3D(T_vec, F_vec, labels, t, x, y, G_raw, G_modal, g, dt, plot_time=0, animate=False, num_frames=100, playback_speed=100):
  T_est_raw, T_est_modal = np.zeros(np.shape(T_vec)), np.zeros(np.shape(T_vec))

  Nx, Ny = len(x), len(y[0])

  # Set up the figure and axes
  fig, ax = plt.subplots(len(T_vec) - 1, 3, figsize=(15, 5 * (len(T_vec) - 1)), sharex=True, sharey=True)
  plt.subplots_adjust(wspace=0.4, hspace=0.4)

  # Prepare X, Y meshgrid
  X, Y = np.meshgrid(x, y)

  # Create a placeholder for the colorbar
  cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

  for i in range(1, len(T_vec)):
    T_est_raw[i - 1, :, :] = convolve(G_raw*dt, F_vec[i].reshape(1,len(F_vec[i])))[:, :len(F_vec[i])]
    T_est_modal[i - 1, :, :] = np.dot(g, convolve(G_modal*dt, F_vec[i].reshape(1,len(F_vec[i])))[:, :len(F_vec[i])])

  # Calculate the global min and max for consistent color mapping
  vmin = min(T_vec[0].min(), T_vec[1].min(), T_vec[2].min())
  vmax = max(T_vec[0].max(), T_vec[1].max(), T_vec[2].max())

  T_vec_0_reshaped = T_vec[0].reshape(Nx, Ny, -1)
  for i in range(1, len(T_vec)):
    # Reshape the vectors from (Nx*Ny, Nt) to (Nx, Ny, Nt)
    T_vec_reshaped = T_vec[i].reshape(Nx, Ny, -1)
    T_est_raw_reshaped = T_est_raw[i - 1].reshape(Nx, Ny, -1)
    T_est_modal_reshaped = T_est_modal[i - 1].reshape(Nx, Ny, -1)

    if not animate:
      # Select the time slice for contour plotting
      time_label = f"t = {t[plot_time]:.2f}"

      # Ensure x and y are properly shaped
      X, Y = np.meshgrid(x, y)

      # Contour plots
      cont0 = ax[i - 1, 0].contourf(X, Y, T_vec_reshaped[:, :, plot_time] - T_vec_0_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i - 1, 0].set_title(f"{labels[i]} - {time_label}")
      cont1 = ax[i - 1, 1].contourf(X, Y, T_est_raw_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i - 1, 1].set_title(f'Recon. {labels[i]}: raw - {time_label}')
      cont2 = ax[i - 1, 2].contourf(X, Y, T_est_modal_reshaped[:, :, plot_time], vmin=vmin, vmax=vmax)
      ax[i - 1, 2].set_title(f'Recon. {labels[i]}: modal - {time_label}')

      # Set labels
      for j in range(3):
        ax[i - 1, j].set_xlabel('x')
        ax[i - 1, j].set_ylabel('y')

  if not animate:
    fig.colorbar(cont0, cax=cbar_ax)
    plt.show()

  def anim_3D(k):
    for i in range(len(T_vec)):
      time_label = f"t = {t[k]:.2f}"

      # Clear the axes to redraw
      ax[i - 1, 0].cla()
      ax[i - 1, 1].cla()
      ax[i - 1, 2].cla()

      # Contour plots
      cont0 = ax[i - 1, 0].contourf(X, Y, T_vec_reshaped[:, :, k] - T_vec_0_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i - 1, 0].set_title(f"{labels[i]} - {time_label}")
      cont1 = ax[i - 1, 1].contourf(X, Y, T_est_raw_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i - 1, 1].set_title(f'Recon. {labels[i]}: raw - {time_label}')
      cont2 = ax[i - 1, 2].contourf(X, Y, T_est_modal_reshaped[:, :, k], vmin=vmin, vmax=vmax)
      ax[i - 1, 2].set_title(f'Recon. {labels[i]}: modal - {time_label}')

      # Set labels
      for j in range(3):
        ax[i - 1, j].set_xlabel('x')
        ax[i - 1, j].set_ylabel('y')

    # Set the colorbar only once (for the first frame)
    if k == 0:
        fig.colorbar(cont0, cax=cbar_ax)

  if animate:
    anim = animation.FuncAnimation(
        fig,
        anim_3D,
        frames=num_frames,          # Number of frames to animate
        interval=playback_speed,    # Interval in milliseconds between frames
        repeat=False
    )
    plt.show()

    return anim

  else:
    L2_raw = calc_L2(T_vec[1:] - T_vec[0], T_est_raw, labels[1:], 'raw')
    L2_modal = calc_L2(T_vec[1:] - T_vec[0], T_est_modal, labels[1:], 'modal')

    return L2_raw, L2_modal


def opt_h_lam_3D(params, T, F, t, m, dt):
  h = params[:m]
  lam = params[m]

  G_opt = h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])

  if all(j == 0 for j in F):
    model = G_opt
  else:
    model = convolve(G_opt*dt, F.reshape(1,len(F)))[:, :len(F)]

  return linalg.norm(T - model)

def apply_response_3D(params, t, m):
  h = params[:m]
  lam = params[m]
  return h[:, np.newaxis]*np.exp(-lam*t)*linalg.pinv(h[np.newaxis, :])