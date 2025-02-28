import numpy as np
import matplotlib.pyplot as plt
import utils_general
from scipy.optimize import minimize

########### Instantiate Advection-Diffusion ##########

def create_adv_diff(noisy=False):
  if noisy:
    pass
  else:
    T_ODE = {}
    g_ODE, a_ODE = {}, {}
    for exp in experiments:
      T_ODE[exp] = utils_general.reconstruct_T_2D(F_all[exp], T_shape, L, ic, dt, gamma=gamma)
      g_ODE[exp], a_ODE[exp] = utils_general.calc_modes_2D(T_ODE[exp])

    utils_general.plot_2D(T_ODE, t_mesh, x_mesh, experiments, 'Noiseless ODE Solutions')

  return T_ODE, g_ODE, a_ODE

def construct_L(ax, mu, dx, n):
    # Create an empty L matrix of size NxN
    L = np.zeros((n, n))

    # Fill in advection (first derivative) part
    for i in range(1, n-1):
        L[i, i-1] = ax / (2 * dx)
        L[i, i+1] = -ax / (2 * dx)

    # Fill in diffusion (second derivative) part
    for i in range(1, n-1):
        L[i, i-1] += mu / (dx ** 2)
        L[i, i]   -= 2 * mu / (dx ** 2)
        L[i, i+1] += mu / (dx ** 2)

    # Periodic boundary conditions
    L[0, -1] = ax / (2 * dx) + mu / (dx ** 2)  # For u[0] interacting with u[N-1]
    L[0, 1] = -ax / (2 * dx) + mu / (dx ** 2)  # For u[0] interacting with u[1]
    L[0, 0] = -2 * mu / (dx ** 2)

    L[-1, -2] = ax / (2 * dx) + mu / (dx ** 2)  # For u[N-1] interacting with u[N-2]
    L[-1, 0] = -ax / (2 * dx) + mu / (dx ** 2)  # For u[N-1] interacting with u[0]
    L[-1, -1] = -2 * mu / (dx ** 2)

    return L

def plot_adv_diff(u, t_mesh, x_mesh, ax, mu, F_label):

  plt.figure(figsize=(10, 6))
  plt.contourf(t_mesh, x_mesh, u)
  plt.colorbar()
  plt.title(rf'2D A-D Profile: $a_x = {ax}$, $\mu = {mu}$, $F = {F_label}$')
  plt.ylabel('Location (m)')
  plt.xlabel('Time (s)')

  return

def adv_diff_2D_periodic(u_ic, ax, mu, dx, dt, N_t, t_mesh, x_mesh, F, F_label, plot_soln = False):
  # Copy ICs so we don't operate on the actual dataset
  u = np.copy(u_ic)

  for k in range(N_t - 1):
    # Update the interior points
    u[1:-1, k+1] = (
      - ax * (dt / (2 * dx)) * (u[2:, k] - u[:-2, k])
      + mu * (dt / dx ** 2) * (u[2:, k] - 2 * u[1:-1, k] + u[:-2, k])
    ) + u[1:-1, k] + F[k]*dt

    # Apply periodic boundary conditions
    # Left boundary (x = 0), wrap around to the last point
    u[0, k+1] = (
      - ax * (dt / (2 * dx)) * (u[1, k] - u[-1, k])
      + mu * (dt / dx ** 2) * (u[1, k] - 2 * u[0, k] + u[-1, k])
    ) + u[0, k] + F[k]*dt

    # Right boundary (x = L), wrap around to the first point
    u[-1, k+1] = (
      - ax * (dt / (2 * dx)) * (u[0, k] - u[-2, k])
      + mu * (dt / dx ** 2) * (u[0, k] - 2 * u[-1, k] + u[-2, k])
    ) + u[-1, k] + F[k]*dt

  # Plot solution (optional)
  if plot_soln:
    plt.figure(figsize=(10, 6))
    plt.contourf(t_mesh, x_mesh, u)
    plt.colorbar()
    plt.title(rf'2D A-D Profile: $a_x = {ax}$, $\mu = {mu}$, $F = {F_label}$')
    plt.ylabel('Location (m)')
    plt.xlabel('Time (s)')

  return u

def calc_dt_2D(ax, mu, dx):
  # Pure diffusion
  if ax == 0 and mu != 0:
    return dx ** 2 / (4*mu)

  # Pure advection
  elif ax != 0 and mu == 0:
    return dx / ax

  # Advection-diffusion
  elif ax !=0 and mu != 0:
    return min(dx ** 2 / (4 * mu), dx / ax)

  else:
    raise ValueError('Both coefficients cannot be zero!')

def init_adv_diff_2D(ax, mu, T0, x, t, x_final, ic, plot_ic = False):
  # Initialize grid: the grid of u(i,k)
  T = np.zeros((len(x), len(t)))

  # Set the initial condition
  T[:, 0] = ic

  # Create mesh
  t_mesh, x_mesh = np.meshgrid(t, x)

  # Plot initial conditions (optional)
  if plot_ic:
    plt.figure(figsize=(10, 6))
    plt.contourf(t_mesh, x_mesh, T)
    plt.colorbar()
    plt.title('AD 2D Initial Condition')
    plt.ylabel('Location (m)')
    plt.xlabel('Time (s)')

  return t_mesh, x_mesh, T

########## Methods for L(x,x') or R(x,t) ##########

def method_1_direct(T_ODE, modal=True):

  G_raw, G_modal, g, a = {}, {}, {}, {}
  for exp in experiments:
    G_raw[exp] = utils_general.reconstruct_T_2D(F_del[exp], T_shape, L, ic, dt, gamma=gamma)
    g[exp], a[exp] = utils_general.calc_modes(G_raw[exp])
    G_modal[exp] = g[exp] @ a[exp]

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'G',
                                               G_raw, G_modal, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments,
                                      soln_type='Method 1 Solutions',
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
                                      experiments,
                                      soln_type='Method 2 Solutions',
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
                                      experiments,
                                      soln_type='Method 3 Solutions',
                                      modal=modal, T_modal=T_modal)

  return T_raw, T_modal, G_raw, G_modal

def method_4_help(T_ODE, m, k, gamma, modal=False, g_ODE={}):

  initial_v = np.random.rand(m * k)  # Flattened eigenvector
  initial_lam = np.random.rand(m)      # Eigenvalues
  initial_params = np.concatenate([initial_v, initial_lam])  # Combine into a single parameter vector
  bounds = [(None, None)] * (m * k) + [(-1, 0)] * m

  res, G = {}, {}
  for exp in experiments:
    if not modal:
      g_ODE[exp] = None
    res[exp] = minimize(utils_general.opt_v_lam_2D,
                            initial_params,
                            args=(T_ODE[exp], F_all[exp], t, m, dt, gamma, g_ODE[exp]),
                            method='L-BFGS-B',
                            bounds=bounds)
    G[exp] = utils_general.apply_v_lam_2D(res[exp].x, t, m, gamma, dt, g_ODE[exp])

    if modal:
      G[exp] = g_ODE[exp] @ G[exp]

  return res, G

def method_4_fit(T_ODE, g_ODE, a_ODE, m, modal=True):

  k = np.shape(T_ODE['2xCO2'])[0]
  gamma = np.ones(k)
  res_raw, G_raw = method_4_help(T_ODE, m, k, gamma)

  #k = np.shape(a_ODE['2xCO2'])[0]
  res_modal, G_modal = method_4_help(a_ODE, m, k, gamma, modal=True, g_ODE=g_ODE)

  T_raw, T_modal = utils_general.estimate_T_2D(T_ODE, F_all,
                                               experiments, t, 'G',
                                               G_raw, G_modal, T0, dt)

  L2_raw, L2_modal = calc_L2_and_plot(T_ODE, T_raw, t,
                                      experiments,
                                      soln_type='Method 4 Solutions',
                                      modal=modal, T_modal=T_modal)
  return T_raw, T_modal, G_raw, G_modal

##########

def calc_L2_and_plot(T_ODE, T_raw, t, experiments, soln_type, modal=True, T_modal=None):

  utils_general.plot_2D(T_raw, t_mesh, x_mesh, experiments, f'{soln_type} - Raw')
  L2_raw = utils_general.calc_L2(T_ODE, T_raw, experiments, None, 'Raw', coupled=True)

  if modal:
    utils_general.plot_2D(T_modal, t_mesh, x_mesh, experiments, f'{soln_type} - Modal')
    L2_modal = utils_general.calc_L2(T_ODE, T_modal, experiments, None, 'Modal', coupled=True)
    return L2_raw, L2_modal

  return L2_raw, None

######

# ODE parameters
ax = 10/5
mu = 3/5
T0 = 0

# Grid parameters
dx = 1
dt = calc_dt_2D(ax, mu, dx)
N_x = 250
N_t = 50
x_final = N_x*dx
t_final = N_t*dt
x = np.arange(0, x_final, dx)
t = np.arange(0, t_final, dt)
t_mesh, x_mesh = np.meshgrid(t, x)

T_shape = np.zeros((N_x, N_t))

# Construct explicit linear operator
L = construct_L(ax, mu, dx, N_x)

# Forcing parameters
## 2xCO2 and 4xCO2 (constant forcing)
F_2xCO2 = utils_general.F_const(t, 1.0)
F_4xCO2 = utils_general.F_const(t, 2.0)

## High Emissions
F_final = 2.6 # (W m^-2)
ts = t_final/2
a_exp = F_final/np.exp(N_t/1.5*dt/ts)
F_exp = utils_general.F_exp(t, a_exp, ts)

## Overshoot
a_over = 4
b_over = t_final*(2/5)
c_over = ts/3
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

gamma = np.ones(np.shape(x))
ic = np.zeros(N_x)
t_mesh, x_mesh, T_init = init_adv_diff_2D(ax, mu, T0, x, t, x_final, ic, plot_ic = False)