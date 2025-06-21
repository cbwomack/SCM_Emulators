from venv import create
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import solve_ivp
import random
import BudykoSellers
import os.path
import pickle

import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from jax.example_libraries import optimizers

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use a LaTeX-compatible serif font
    "font.serif": ["Computer Modern Roman"],  # Or another LaTeX font
})

####################################
## Functions to generate forcings ##
####################################

def abrupt_2xCO2(n_boxes, n_t):
  F0 = 3.7 # [W m-2]
  return F0*np.ones((n_boxes, n_t))

def high_emissions(RF_final, t, t_end, t_star, n_boxes):
  a = RF_final/np.exp(t_end/t_star)
  F = a*np.exp(t/t_star)
  return np.tile(F, (n_boxes,1))

def overshoot(t, n_boxes):
  a = 4 # [W m-2]
  b = 200 # [years]
  c = 42.47 # [growth rate]
  F = a*np.exp(-np.power(t - b,2)/(2*c**2))
  return np.tile(F, (n_boxes,1))

def generate_forcing(exp, t, t_end, t_star, n_boxes):
  if exp == 'Abrupt':
    n_t = len(t)
    F = abrupt_2xCO2(n_boxes, n_t)

  elif exp == 'High Emissions':
    F = high_emissions(8.5, t, t_end, t_star, n_boxes)

  elif exp == 'Mid. Emissions':
    F = high_emissions(4.5, t, t_end, t_star, n_boxes)

  elif exp == 'Overshoot':
    F = overshoot(t, n_boxes)

  return F

########################################
## Quasi-Equilibrium Helper Functions ##
########################################

def generate_QE(exp, F0=0, F1=10, N_step=100):
  if exp not in ['3box_coup','2box_coup','lorenz']:
    raise ValueError(f'Error: experiment {exp} not recognized.')

  file_path = f'Quasi-Equilibrium/{exp}.pkl'
  if os.path.isfile(file_path):
    raise ValueError(f'Error: QE lookup table already exists for experiment {exp}.')

  F_range = np.linspace(F0, F1, N_step)
  QE_lookup = {}

  for F in F_range:
    if exp == '3box_coup':
      full_outputs = BudykoSellers.Run_Budyko_Sellers(scen_flag=0, n_boxes=3, diff_flag=1, F0=F, int_yrs=1000)
      T_out = np.squeeze(full_outputs['T_ts'])[0:3,:]
      QE_lookup[F] = T_out[:,-1]

  with open(file_path, 'wb') as file:
    pkl.dump(QE_lookup, file)

  return QE_lookup

def load_QE(exp):

  file_path = f'Quasi-Equilibrium/{exp}.pkl'
  with open(file_path, 'rb') as file:
    QE_lookup = pkl.load(file)

  return QE_lookup

def QE_interp(QE_lookup, F_vec, n_boxes):

  if F_vec.ndim == 2:
    F_vec = F_vec[0]

  F_eq = np.array(sorted(QE_lookup))
  T_eq = np.vstack([QE_lookup[k] for k in F_eq])

  T_interp = np.zeros((n_boxes, len(F_vec)))

  for i, F in enumerate(F_vec):
    if F in QE_lookup:
      T_interp[:,i] = QE_lookup[F]
    else:

      idx = np.searchsorted(F_eq, F)
      if idx == 0:
        i0, i1 = 0, 1
      elif idx == len(F_eq):
        i0, i1 = -2, -1
      else:
        i0, i1 = idx-1, idx

    F0, F1 = F_eq[i0], F_eq[i1]
    w = (F - F0) / (F1 - F0)

    # Linear interpolation component-wise
    T_interp[:,i] = (1 - w) * T_eq[i0] + w * T_eq[i1]

  return T_interp

######################################
## Functions for Lorenz-like System ##
######################################

def Lorenz_rho(t, omega=1, offset=10, exp=0):
  # Calculate rho for Lorenz-like system
  if exp == 0: # Abrupt
    return 45 + 17*np.tanh(omega*(t - 10))
  elif exp == 1: # High Emissions
    return 28 + 30/(np.exp(250/50))*np.exp(t/50)
  elif exp == 2: # Mid. Emissions
    return 40 + 12*np.tanh(1/50*(t - 150))/np.tanh(5)
  elif exp == 3: # Overshoot
    return 28 + 30*np.exp(-np.power(t - 200,2)/(2*50**2))
  elif exp == 4: # Sinusoid
    return 60 + 30*np.sin(omega*t)
  elif exp == 5: # Noise
    return (0.25*np.exp(-t/5) + 0.75*np.exp(-t/0.05))*np.cos(2*np.pi*t/0.57)
  elif exp == 6:
    return 30 # Constant initial condition
  else:
    raise ValueError('Error, unrecognized experiment.')

def Lorenz_ddt(t, state_vec_flat, alpha, beta, sigma, omega, offset, exp=0, FDT=False, pert=0, dt_imp=None):
  state_vec = state_vec_flat.reshape(-1, 3)

  rho = Lorenz_rho(t, omega, offset, exp)
  if FDT and t < dt_imp:
    rho += pert

  x, y, z = state_vec[:,0], state_vec[:,1], state_vec[:,2]
  dx = sigma*(y - x)
  dy = -(z + alpha*np.pow(z,3))*x + rho*x - y
  dz = x*y - beta*z

  ddt_flat = np.column_stack([dx, dy, dz]).ravel()

  return ddt_flat

def Lorenz_init(t_max, dt, N, r, alpha, beta, sigma, omega, offset=50, exp=6):

  rho_0 = Lorenz_rho(exp, omega, offset, exp)
  x0 = 2.0 * np.random.rand(N) - 1.0
  y0 = 2.0 * np.random.rand(N) - 1.0
  z0 = (rho_0 - 1) * np.ones(N)

  # Combine x,y,z into one array of shape (N, 3)
  state_vec = np.column_stack([x0, y0, z0])

  t = 0
  while t < t_max:
    # Single-step ODE from t to t+dt using solve_ivp with method=RK45
    sol = solve_ivp(Lorenz_ddt, [t, t + dt], state_vec.flatten(),
                    method='RK45', t_eval=[t + dt],
                    args=(alpha, beta, sigma, omega, offset, exp))

    # Extract the final state, reshape it back to (N,3)
    state_vec = sol.y[:, -1].reshape((N, 3))
    t += dt

    # Add the Gaussian noise
    noise = r * np.random.randn(*state_vec.shape)
    state_vec += noise

  # Store statistics
  x = state_vec[:, 0]
  y = state_vec[:, 1]
  z = state_vec[:, 2]

  return x, y, z

def Lorenz_FDT(N, dt, t_max):
  # Params
  sigma, beta = 10.0, 8 / 3
  rho_base = 30.0
  delta_rho = 1.0

  D = 1.0
  alpha = 1/1000
  t = np.arange(0.0, t_max + dt, dt)
  n_steps = t.size
  sqrt_2Ddt = np.sqrt(2 * D * dt)

  def lorenz(state, rho):
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    dx = sigma * (y - x)
    dy = -(z + alpha*np.pow(z,3))*x + rho*x - y
    dz = x * y - beta * z
    return np.stack([dx, dy, dz], axis=1)

  # Spin up
  state0 = np.random.normal(0.0, 5.0, size=(N, 3))
  warm_steps = 1000
  for _ in range(warm_steps):
    dW = np.random.normal(0.0, 0.1, size=state0.shape)
    state0 += lorenz(state0, rho_base) * dt + sqrt_2Ddt * dW

  # Baseline and perturbed ensembles share identical initial states
  state_base = state0.copy()
  state_pert = state0.copy()

  # Store means
  mean_base = np.zeros((n_steps, 3))
  mean_pert = np.zeros_like(mean_base)
  mean_base[0] = state_base.mean(axis=0)
  mean_pert[0] = state_pert.mean(axis=0)

  # Integrate
  t_check = 0
  for n in range(1, n_steps):
    dW = np.random.normal(0.0, 0.1, size=state_base.shape)
    state_base += lorenz(state_base, rho_base) * dt + sqrt_2Ddt * dW
    if t_check < 1:
      state_pert += lorenz(state_pert, rho_base + delta_rho) * dt + sqrt_2Ddt * dW
    else:
      state_pert += lorenz(state_pert, rho_base) * dt + sqrt_2Ddt * dW

    mean_base[n] = state_base.mean(axis=0)
    mean_pert[n] = state_pert.mean(axis=0)
    t_check += dt

  # FDT
  R = (mean_pert - mean_base) / delta_rho

  return R

def Lorenz_integrate(t_max, dt, N, r, alpha, beta, sigma, omega, offset, x0, y0, z0, exp=0, FDT=False):

  if N < 50:
    n_snap = 1
  else:
    n_snap = N

  # Initial conditions
  rho_0 = Lorenz_rho(0, omega, offset, exp)

  # Combine x,y,z into one array of shape (N, 3)
  state_vec = np.column_stack([x0, y0, z0])

  # Pre-allocate arrays for storage
  nt = int(t_max / dt) + 2
  x_snap   = np.zeros((n_snap, nt))   # snapshot of first 50 x-values
  y_snap   = np.zeros((n_snap, nt))   # snapshot of first 50 y-values
  z_snap   = np.zeros((n_snap, nt))   # snapshot of first 50 z-values
  z_mean   = np.zeros(nt)         # mean of z-component
  z_max   = np.zeros((3, nt))    # max of each column in xv
  z_std = np.zeros(nt)         # std of z-component
  rho = np.zeros(nt)

  # Fill initial values
  j = 0
  x_snap[:, j] = state_vec[:n_snap, 0]
  y_snap[:, j] = state_vec[:n_snap, 1]
  z_snap[:, j] = state_vec[:n_snap, 2]
  z_mean[j]   = np.mean(state_vec[:, 2])
  z_max[:, j] = np.max(state_vec, axis=0)
  z_std[j]  = np.std(state_vec[:, 2])
  rho[j] = rho_0

  t = 0
  while t < t_max:
    # Single-step ODE from t to t+dt using solve_ivp with method=RK45
    sol = solve_ivp(Lorenz_ddt, [t, t + dt], state_vec.flatten(),
                    method='RK45', t_eval=[t + dt],
                    args=(alpha, beta, sigma, omega, offset, exp))

    # Extract the final state, reshape it back to (N,3)
    state_vec = sol.y[:, -1].reshape((N, 3))
    t += dt

    # Add the Gaussian noise
    noise = r * np.random.randn(*state_vec.shape)
    state_vec += noise

    # Store statistics
    j += 1
    x_snap[:, j] = state_vec[:n_snap, 0]
    y_snap[:, j] = state_vec[:n_snap, 1]
    z_snap[:, j] = state_vec[:n_snap, 2]
    z_mean[j]   = np.mean(state_vec[:, 2])
    z_max[:, j] = np.max(state_vec, axis=0)
    z_std[j]  = np.std(state_vec[:, 2])
    rho[j] = Lorenz_rho(t, omega, offset, exp)

  return x_snap, y_snap, z_snap, z_mean, z_max, z_std, rho

def Lorenz_plot(Z_mean, Z_std, t_vec, offset, T):
  i_mid = np.where((t_vec >= offset - 3*T) & (t_vec <= offset + 3*T))[0]

  fig, ax = plt.subplots(2, 1, figsize=(10,5), constrained_layout=True)
  ax[0].plot(t_vec, Z_mean, c=brewer2_light(0), lw=2, label='mean')
  ax[0].fill_between(t_vec, Z_mean - Z_std, Z_mean + Z_std, alpha=0.5, color=brewer2_light(0), label=r'$\sigma$')
  ax[0].legend()

  ax[1].plot(t_vec[i_mid], Z_mean[i_mid], c=brewer2_light(0), lw=2)
  ax[1].fill_between(t_vec[i_mid], Z_mean[i_mid] - Z_std[i_mid], Z_mean[i_mid] + Z_std[i_mid], alpha=0.5, color=brewer2_light(0))

  fig.suptitle(f'Cubic Lorenz System, Transition Time = {T}')

  return

#####################################
## Functions to Diagnose Emulators ##
#####################################

def check_dim(var, transp=False):
  if var.ndim == 1:
    if transp:
      return var.reshape(-1, 1)
    else:
      return var.reshape(1, -1)
  else:
    return var

def method_0_PS(w):
  nx = len(w)
  global_mean = np.mean(w,axis=0).reshape(-1, 1)
  XTX_inv = np.linalg.inv(global_mean.T @ global_mean)  # shape (1, 1)
  XTY = global_mean.T @ w.T  # shape (1, nx)
  pattern = (XTX_inv @ XTY).reshape(1, nx)  # shape (1, nx)

  return pattern

import numpy as np


def method_0b_gen_PS(w, F):
  F = np.asarray(F)
  F = F.reshape(1, -1)
  w = np.asarray(w)

  if w.ndim == 1:            # single grid‐point case
    w = w.reshape(1, -1)
  elif w.shape[0] == F.shape[1] and w.shape[1] != F.shape[1]:
    # transpose to (Nx, Nt)
    w = w.T

  if w.shape[1] != F.shape[1]:
    raise ValueError("Time dimension of w must match that of F")

  XtX = F @ F.T                    # shape (1, 1)
  pattern = (1.0 / XtX) @ (F @ w.T)   # shape (1, Nx)

  return pattern

def method_1a_DMD(w, F):
  # Calculate L using DMD
  # Assume F is of size (nx, nt)
  w, F = check_dim(w), check_dim(F)
  Omega = np.concatenate([w[:,:-1],F[:,:-1]])
  L = w[:,1:] @ np.linalg.pinv(Omega)
  n = len(w)

  A_DMD, B_DMD = L[:,:n], L[:,n:]

  return A_DMD, B_DMD


def method_1b_EDMD(w, F, dict_w, dict_F):
  # Calculate K using EDMD
  w, F = check_dim(w), check_dim(F)
  F = F.T

  Phi_F = dict_F.transform(F[:-1,:])
  Phi_w = dict_w.transform(w[:,:-1].T)
  Phi_wprime = dict_w.transform(w[:,1:].T)

  Omega = np.concatenate([Phi_w.T,Phi_F.T])
  K = Phi_wprime.T @ np.linalg.pinv(Omega)
  n = len(Phi_wprime.T)

  A_EDMD, B_EDMD = K[:,:n], K[:,n:]

  return A_EDMD, B_EDMD

def method_2a_direct(n_boxes, diff_flag=0, vert_diff_flag=0, xi=0, spatial_flag=0, delta=0):
  # Calculate G directly
  full_output_unpert = BudykoSellers.Run_Budyko_Sellers(scen_flag=4, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, spatial_flag=spatial_flag)
  full_output_pert = BudykoSellers.Run_Budyko_Sellers(scen_flag=4, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, spatial_flag=spatial_flag, delta=delta)
  G_direct = (np.squeeze(full_output_pert['T_ts'])[0:n_boxes,:] - np.squeeze(full_output_unpert['T_ts'])[0:n_boxes,:])/delta

  return G_direct

def method_2b_FDT(n_ensemble, n_boxes, n_steps, xi, delta, scen_flag=0, diff_flag=0, vert_diff_flag=0):
  # Calculate G from an ensemble using the FDT
  w, w_delta = np.zeros((n_ensemble, n_boxes, n_steps)), np.zeros((n_ensemble, n_boxes, n_steps))

  # Run n_ensemble number of ensemble members
  for n in range(n_ensemble):
    # Run unperturbed scenario
    full_output_unperturbed = BudykoSellers.Run_Budyko_Sellers(scen_flag=scen_flag, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi)
    w[n,:,:] = np.squeeze(full_output_unperturbed['T_ts'])[0:n_boxes,:]
    noise_ts = full_output_unperturbed['noise_ts']

    # Run perturbed scenario
    full_output_perturbed = BudykoSellers.Run_Budyko_Sellers(scen_flag=scen_flag, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, delta=delta, noise_ts=noise_ts)
    w_delta[n,:,:] = np.squeeze(full_output_perturbed['T_ts'])[0:n_boxes,:]

  # Take ensemble average divided by magnitude of perturbation
  G_FDT = np.mean(w_delta - w, axis=0)/delta

  return G_FDT

def method_3a_deconvolve(w, F, dt, regularize=False):
  # Calculate G using deconvolution
  F = check_dim(F)
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  if regularize:
    sig2, lam2 = get_regularization(w, F)
    alpha = sig2 / lam2  # Ridge penalty
    G_deconv = np.linalg.solve(F_toep.T @ F_toep + alpha * np.eye(F.shape[1]), F_toep.T @ w.T)

  else:
    G_deconv = spsolve_triangular(F_toep, w.T, lower=True)

  return G_deconv.T/dt

"""
def method_3b_Edeconvolve(w, F, dict_w, dict_F):
  # Calculate G using extended deconvolution

  return G_Edeconv
"""
def method_4a_fit(w, F, t, dt, n_modes, n_boxes, B, A_DMD=None, B_DMD=None):
  # Calculate G using an exponential fit
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  if A_DMD is not None:
    eigs, vecs = np.linalg.eig(A_DMD)
    initial_v = -vecs[0:n_modes].flatten()
    initial_lam = -eigs[0:n_modes]

  else:
    initial_v = np.random.rand(n_modes * n_boxes)  # Flattened eigenvector
    initial_lam = np.random.rand(n_modes)      # Eigenvalues

  initial_params = np.concatenate([initial_v, initial_lam])  # Combine into a single parameter vector
  bounds = [(None, None)] * (n_modes * n_boxes) + [(None, 0)] * n_modes

  res = minimize(fit_opt_eigs,
                 initial_params,
                 args=(w, F_toep, t, dt, B, n_modes, n_boxes),
                       method='L-BFGS-B',
                       bounds=bounds)

  debug = True
  if debug:
    print('Exponential Fit Results:\n',res.x)

  return fit_opt_eigs(res.x, w, F_toep, t, dt, B, n_modes, n_boxes, apply=True)

def method_4b_fit_jax(w, F, t, dt, n_modes, n_boxes, B, num_steps=100, verbose=False):

  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  F_toep = jnp.array(F_toep.toarray())
  dt = jnp.array(dt)
  w = jnp.array(w)
  B = jnp.array(B)
  t = jnp.array(t)

  def make_fit_opt_eigs_jax(n_modes: int, n_boxes: int):
    """Return a function that knows n_modes, n_boxes as Python ints."""
    def fit_opt_eigs_jax(params, w, F_toep, t, dt, B):
      alpha = params[:n_modes * n_boxes].reshape((n_boxes, n_modes))
      v = jnp.exp(alpha)
      v_inv = jnp.linalg.pinv(v)
      theta = params[n_modes * n_boxes : n_modes * n_boxes + n_modes]
      lam = -jnp.exp(theta)

      G_opt = jnp.zeros((n_boxes, len(t)))
      for i, time_val in enumerate(t):
        if i == 0:
          continue
        exp_diag_trunc = jnp.diag(jnp.exp(lam * time_val))
        exp_Lt_trunc = v @ exp_diag_trunc @ v_inv
        G_opt = G_opt.at[:, i].set(exp_Lt_trunc @ B)

      model = (G_opt * dt) @ F_toep.T
      return jnp.linalg.norm(w - model, ord=2)
    return fit_opt_eigs_jax

  initial_v = jax.random.normal(jax.random.PRNGKey(0), (n_modes*n_boxes,))
  initial_theta = jax.random.normal(jax.random.PRNGKey(1), (n_modes,))
  initial_params = jnp.concatenate([initial_v, initial_theta])

  fit_fn = make_fit_opt_eigs_jax(n_modes=n_modes, n_boxes=n_boxes)
  def cost_fn(params, w, F_toep, t, dt, B):
    """
    Cost function that calls fit_fn with your data.
    """
    return fit_fn(params, w, F_toep, t, dt, B)

  learning_rate = 0.1
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)

  @jax.jit
  def update(step, opt_state, w, F_toep, t, dt, B):
      params = get_params(opt_state)
      # Compute gradients wrt. the cost
      grads = jax.grad(cost_fn)(params, w, F_toep, t, dt, B)
      # Apply the optimizer step
      new_opt_state = opt_update(step, grads, opt_state)
      return new_opt_state
  # Initialize the optimizer with the initial parameters
  opt_state = opt_init(initial_params)

  for step_i in range(1000):
    opt_state = update(step_i, opt_state, w, F_toep, t, dt, B)
    if step_i % 100 == 0 and True:
      # Optional: check cost
      current_params = get_params(opt_state)
      cval = cost_fn(current_params, w, F_toep, t, dt, B)
      print(f"Step {step_i}, cost={cval:0.6f}")

  opt_params = get_params(opt_state)
  opt_params_trans = theta_to_lam(opt_params, n_modes, n_boxes)

  if True:
    print(opt_params_trans)

  return fit_opt_eigs(opt_params_trans, w, F_toep, t, dt, B, n_modes, n_boxes, apply=True)

def theta_to_lam(params, n_modes, n_boxes):
  alpha = params[:n_modes * n_boxes]
  v = jnp.exp(alpha)
  theta = params[n_modes * n_boxes : n_modes * n_boxes + n_modes]
  params = jnp.concatenate([v, -jnp.exp(theta)])
  return params

def method_4c_fit_amp(w, F, t, dt, n_modes, n_boxes, num_steps=100, verbose=False):

  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  F_toep = jnp.array(F_toep.toarray())
  dt = jnp.array(dt)
  w = jnp.array(w)
  t = jnp.array(t)

  def make_fit_opt_eigs_jax(n_modes: int, n_boxes: int):
    def fit_opt_eigs_amp(params, w, F_toep, t, dt):
      #beta = params[:n_boxes]
      #theta = params[n_boxes:]
      beta = params[:n_boxes*n_modes].reshape((n_boxes, n_modes))
      theta = params[n_boxes*n_modes:].reshape((n_modes, 1))
      alpha = jnp.exp(beta)
      #alpha = beta
      alpha_diag = 1.0 / jnp.exp(jnp.diagonal(beta))
      alpha = alpha.at[jnp.diag_indices(n_boxes)].set(alpha_diag)
      lam = -jnp.exp(theta)
      G_opt = jnp.zeros((n_boxes, len(t)))

      for i, time_val in enumerate(t):
        if i == 0:
          continue
        #exp_Lt_trunc = (1/alpha*jnp.exp(lam*time_val)).reshape(n_boxes)
        exp_Lt_trunc = (alpha @ jnp.exp(lam*time_val)).reshape(n_boxes)
        G_opt = G_opt.at[:, i].set(exp_Lt_trunc)

      model = (G_opt * dt) @ F_toep.T
      l1_penalty = 100*jnp.sum(jnp.abs(alpha * (1.0 - jnp.eye(alpha.shape[0]))))
      return jnp.linalg.norm(w - model, ord=2) + l1_penalty
    return fit_opt_eigs_amp

  key1, key2 = np.random.normal(size=2)
  #initial_phi = jax.random.normal(jax.random.PRNGKey(0), (n_boxes,))
  #initial_phi = jax.random.normal(jax.random.PRNGKey(0), (n_modes*n_boxes,))
  #initial_theta = jax.random.normal(jax.random.PRNGKey(1), (n_modes,))

  if n_modes == 3:
    initial_phi = np.log(np.array([1e2,1e-3,1e-3,1e-3,1,1e-3,1e-3,1e-3,1e1]))
    initial_theta = np.log(np.array([1e-3,1e-1,1e-2]))
  elif n_boxes == 1 and n_modes == 2:
    initial_phi = np.log(np.array([1e2,1e-3]))
    initial_theta = np.log(np.array([1e-3,1e-1]))
  initial_params = jnp.concatenate([initial_phi, initial_theta])

  fit_fn = make_fit_opt_eigs_jax(n_modes=n_modes, n_boxes=n_boxes)
  def cost_fn(params, w, F_toep, t, dt):
    """
    Cost function that calls fit_fn with your data.
    """
    return fit_fn(params, w, F_toep, t, dt)

  learning_rate = 0.1
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)

  @jax.jit
  def update(step, opt_state, w, F_toep, t, dt):
    params = get_params(opt_state)
    grads = jax.grad(cost_fn)(params, w, F_toep, t, dt)
    new_opt_state = opt_update(step, grads, opt_state)
    return new_opt_state

  opt_state = opt_init(initial_params)

  for step_i in range(1000):
    opt_state = update(step_i, opt_state, w, F_toep, t, dt)
    if step_i % 100 == 0 and False:
      current_params = get_params(opt_state)
      cval = cost_fn(current_params, w, F_toep, t, dt)
      print(f"Step {step_i}, cost={cval:0.6f}")

  opt_params = get_params(opt_state)

  if verbose:
    print(opt_params)

  return apply_amp(opt_params, t, n_modes, n_boxes)

def apply_amp(params, t, n_modes, n_boxes):
  #beta = params[:n_boxes]
  #theta = params[n_boxes:]
  beta = params[:n_boxes*n_modes].reshape((n_boxes, n_modes))
  theta = params[n_boxes*n_modes:].reshape((n_modes, 1))
  alpha = jnp.exp(beta)
  #alpha = beta
  alpha_diag = 1.0 / jnp.exp(jnp.diagonal(beta))
  alpha = alpha.at[jnp.diag_indices(n_boxes)].set(alpha_diag)
  lam = -jnp.exp(theta)

  G_opt = np.zeros((n_boxes, len(t)))

  for i, n in enumerate(t):
    if i == 0:
      continue
    #G_opt[:,i] = 1/alpha * jnp.exp(lam*n)
    G_opt[:,i] = (alpha @ jnp.exp(lam*n)).reshape(n_boxes)

  return G_opt

def method_4d_fit_complex_old(w, F, t, dt, n_modes, n_boxes, num_steps=500, verbose=False):

  F = check_dim(F)
  w = check_dim(w)

  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  F_toep = jnp.array(F_toep.toarray())
  dt = jnp.array(dt)
  w = jnp.array(w)
  t = jnp.array(t)

  def make_fit_opt_eigs_jax(n_modes: int, n_boxes: int):
    def fit_opt_eigs_amp(params, w, F_toep, t, dt):
      phi = params[:n_boxes]
      theta = params[n_boxes:2*n_boxes]
      beta = params[2*n_boxes:]
      alpha = jnp.exp(phi)
      lam = -jnp.exp(theta)
      omega = jnp.exp(beta)
      G_opt = jnp.zeros((n_boxes, len(t)))

      for i, time_val in enumerate(t):
        if i == 0:
          continue
        exp_Lt_trunc = (1/alpha*jnp.exp(1/alpha * (lam + 1j*omega)*time_val)).reshape(n_boxes)
        G_opt = G_opt.at[:, i].set(exp_Lt_trunc)

      model = (G_opt * dt) @ F_toep.T
      return jnp.linalg.norm(w - model, ord=2)
    return fit_opt_eigs_amp

  key1, key2 = np.random.normal(size=2)
  initial_phi = jax.random.normal(jax.random.PRNGKey(0), (n_modes,))
  initial_theta = jax.random.normal(jax.random.PRNGKey(1), (n_modes,))
  initial_beta = jax.random.normal(jax.random.PRNGKey(2), (n_modes,))
  initial_params = jnp.concatenate([initial_phi, initial_theta, initial_beta])

  fit_fn = make_fit_opt_eigs_jax(n_modes=n_modes, n_boxes=n_boxes)
  def cost_fn(params, w, F_toep, t, dt):
    """
    Cost function that calls fit_fn with your data.
    """
    return fit_fn(params, w, F_toep, t, dt)

  learning_rate = 0.1
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)

  @jax.jit
  def update(step, opt_state, w, F_toep, t, dt):
    params = get_params(opt_state)
    grads = jax.grad(cost_fn)(params, w, F_toep, t, dt)
    new_opt_state = opt_update(step, grads, opt_state)
    return new_opt_state

  opt_state = opt_init(initial_params)

  for step_i in range(100):
    opt_state = update(step_i, opt_state, w, F_toep, t, dt)
    if step_i % 10 == 0 and True:
      current_params = get_params(opt_state)
      cval = cost_fn(current_params, w, F_toep, t, dt)
      print(f"Step {step_i}, cost={cval:0.6f}")

  opt_params = get_params(opt_state)

  if verbose:
    print(opt_params)

  return apply_complex(opt_params, t, n_modes, n_boxes)



def method_4d_fit_complex(w, F, t, dt, n_modes, n_boxes, num_steps=500, learning_rate=0.1, verbose=False):

    # --- 1. Data Preparation ---
    # This part remains mostly the same
    F = check_dim(F)
    w = check_dim(w)

    F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
    F_toep_T = jnp.array(F_toep.T.toarray()) # Pre-calculate transpose and convert to JAX array
    dt = jnp.array(dt)
    w = jnp.array(w)
    t = jnp.array(t)

    # --- 2. Vectorized Cost Function ---
    # The nested factory function is removed for clarity.
    # The core logic is now directly in cost_fn.
    def cost_fn(params, w_target, t_vec, F_matrix_T):
        phi = params[:n_boxes]
        theta = params[n_boxes:2*n_boxes]
        beta = params[2*n_boxes:]

        alpha = jnp.exp(phi)
        lam = -jnp.exp(theta)
        omega = jnp.exp(beta)

        # KEY OPTIMIZATION: Vectorize the loop over time `t` using broadcasting.
        # This replaces the slow Python `for` loop.
        alpha_r = alpha[:, jnp.newaxis]
        lam_r = lam[:, jnp.newaxis]
        omega_r = omega[:, jnp.newaxis]
        t_r = t_vec[jnp.newaxis, :]

        # G_opt is now calculated in a single, fast operation
        G_opt = (1/alpha_r) * jnp.exp((1/alpha_r) * (lam_r + 1j*omega_r) * t_r)

        model = (G_opt * dt) @ F_matrix_T
        return jnp.linalg.norm(w_target - model, ord=2)

    # --- 3. JIT-Compiled Training Step (Optimized) ---
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)

    @jax.jit
    def update_step(step, opt_state, w_target, t_vec, F_matrix_T):
        # Get current parameters from the optimizer state
        params = get_params(opt_state)
        
        # KEY OPTIMIZATION: Calculate loss and gradients at the same time
        # to avoid redundant forward passes.
        loss_value, grads = jax.value_and_grad(cost_fn)(params, w_target, t_vec, F_matrix_T)
        
        # Update the optimizer state
        new_opt_state = opt_update(step, grads, opt_state)
        
        return new_opt_state, loss_value

    # --- 4. Initialization and Training Loop ---
    # Proper JAX-style random key initialization
    key = jax.random.PRNGKey(42)
    key, phi_key, theta_key, beta_key = jax.random.split(key, 4)
    initial_phi = jax.random.normal(phi_key, (n_modes,))
    initial_theta = jax.random.normal(theta_key, (n_modes,))
    initial_beta = jax.random.normal(beta_key, (n_modes,))
    initial_params = jnp.concatenate([initial_phi, initial_theta, initial_beta])

    # Initialize optimizer
    opt_state = opt_init(initial_params)

    # Cleaned-up training loop
    for step_i in range(num_steps):
        # The update function now returns the loss value directly
        opt_state, cval = update_step(step_i, opt_state, w, t, F_toep_T)
        
        if False:#step_i % 10 == 0:
            print(f"Step {step_i}, cost={cval:0.6f}")

    # Get final parameters after training
    opt_params = get_params(opt_state)

    if verbose:
        print("Final optimized parameters:", opt_params)

    return apply_complex(opt_params, t, n_modes, n_boxes)





def apply_complex(params, t, n_modes, n_boxes):
  phi = params[:n_boxes]
  theta = params[n_boxes:2*n_boxes]
  beta = params[2*n_boxes:]
  alpha = jnp.exp(phi)
  lam = -jnp.exp(theta)
  omega = jnp.exp(beta)

  G_opt = np.zeros((n_boxes, len(t)))

  for i, n in enumerate(t):
    if i == 0:
      continue
    G_opt[:,i] = 1/alpha*jnp.exp(1/alpha*(lam + 1j*omega)*n)

  return G_opt

def fit_opt_eigs(params, w, F_toep, t, dt, B, n_modes, n_boxes, apply=False):
  # Extract eigenvectors (flattened into a 1D vector) and eigenvalues from the parameter vector
  # The first m*n values correspond to the eigenvectors (flattened)
  v = params[:n_modes * n_boxes].reshape(n_boxes, n_modes)  # Reshape to a matrix of size n x m
  v_inv = np.linalg.pinv(v)                     # Compute the pseudo-inverse of v (v^-1)
  lam = params[n_modes * n_boxes:n_modes * n_boxes + n_modes]     # The next m values are the eigenvalues

  G_opt = np.zeros((n_boxes, len(t)))

  for i, n in enumerate(t):
    if i == 0:
      continue
    # Create diagonal matrix of exp(lambda_i * t) for this particular time t
    exp_diag_trunc = np.diag(np.exp(lam * n))

    # Compute exp(L * t) using eigenvalue expansion: v * exp(diag(lambda) * t) * v^-1
    exp_Lt_trunc = v @ exp_diag_trunc @ v_inv

    # Compute G(x, t) by multiplying exp(L * t) with B
    G_opt[:, i] = np.dot(exp_Lt_trunc, B)
    model = (G_opt * dt) @ F_toep.T

  if apply:
    return G_opt
  return np.linalg.norm(w - model)

def create_emulator(op_type, w, F, t=None, dt=None, n_boxes=None, w_dict=None, F_dict=None, n_modes=None, diff_flag=0,
                    vert_diff_flag=0, B=None, xi=0, n_ensemble=None, n_steps=None, delta=0, regularize=False, verbose=False,
                    spatial_flag=0, exp=None):
  if op_type == 'DMD':
    A_DMD, B_DMD = method_1a_DMD(w, F)
    operator = (A_DMD, B_DMD)

  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = method_1b_EDMD(w, F, w_dict, F_dict)
    operator = (A_EDMD, B_EDMD)

  elif op_type == 'deconvolve':
    operator = method_3a_deconvolve(w, F, dt, regularize)

  elif op_type == 'direct':
    operator = method_2a_direct(n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, spatial_flag=spatial_flag, delta=delta)

  elif op_type == 'FDT':
    operator = method_2b_FDT(n_ensemble, n_boxes, n_steps,  xi, delta, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag)

  elif op_type == 'fit':
    operator = method_4a_fit(w, F, t, dt, n_modes, n_boxes, B)

  elif op_type == 'fit_DMD':
    A_DMD, B_DMD = method_1a_DMD(w, F)
    operator = method_4a_fit(w, F, t, dt, n_modes, n_boxes, B, A_DMD, B_DMD)

  elif op_type == 'fit_jax':
    operator = method_4b_fit_jax(w, F, t, dt, n_modes, n_boxes, B, num_steps=100, verbose=verbose)

  elif op_type == 'fit_amp':
    operator = method_4c_fit_amp(w, F, t, dt, n_modes, n_boxes, num_steps=100, verbose=verbose)

  elif op_type == 'fit_complex':
    operator = method_4d_fit_complex(w, F, t, dt, n_modes, n_boxes, num_steps=100, verbose=verbose)

  elif op_type == 'PS':
    operator = method_0_PS(w)

  elif op_type == 'QE':
    operator = load_QE(exp)

  else:
    raise ValueError(f'Operator type {op_type} not recognized.')

  return operator

########################
## Emulate a Scenario ##
########################

def emulate_PS(w, pattern):
  # Emulate a scenario with pattern scaling
  global_mean = np.mean(w, axis=0).reshape(1,-1)
  w_pred = pattern.T @ global_mean

  return w_pred

def emulate_DMD(F, A_DMD, B_DMD, w0, n_steps):
  # Emulate a scenario with DMD
  F = check_dim(F)
  w_pred = np.zeros((w0.shape[0], n_steps))
  w_pred[:, 0] = w0

  for k in range(1, n_steps):
    w_pred[:, k] = A_DMD @ w_pred[:, k-1] + B_DMD @ F[:,k-1]

  return w_pred

def emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F):
  # x0 is shape (3,)
  # Convert to (1,3) for dictionary.transform
  F = check_dim(F)
  phi0 = dict_w.transform(w0.reshape(1, -1))  # shape (1, n_lifted)
  phi0 = phi0.flatten()                            # (n_lifted,)

  # Allocate
  phi_pred = np.zeros((phi0.shape[0], n_steps))
  phi_pred[:, 0] = phi0

  # For storing the reconstructed state
  w_rec = np.zeros((n_boxes, n_steps))

  # We'll fill first step
  w_rec[:, 0] = w0

  F = F.T
  #F = np.hstack((F.T, np.zeros((F.shape[1], 2))))
  Phi_F = dict_F.transform(F[:-1,:]).T

  for k in range(1, n_steps):
    # Discrete-time update in lifted space
    phi_pred[:, k] = A_EDMD @ phi_pred[:, k-1] + B_EDMD @ Phi_F[:,k-1]

    # Reconstruct the state from the first n_boxes dimensions
    #w_rec[:, k] = phi_pred[1:n_boxes+1, k]  # Depends on how dictionary is defined
    w_rec[:, k] = phi_pred[0:n_boxes, k]  # Depends on how dictionary is defined

  return w_rec

def emulate_response(F, G, dt):
  # Emulate a scenario with a response function
  if F.ndim == 1:
    F = F.reshape(1, -1)
  F_toeplitz = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  return G @ F_toeplitz.T * dt

#####################################
## Functions to Evaluate Emulators ##
#####################################

def estimate_w(F, operator, op_type, dt=None, w0=None, n_steps=None, n_boxes=None, dict_w=None, dict_F=None, w=None):
  # Estimate variable of interest given an initial
  # condition and forcing

  if op_type == 'DMD':
    A_DMD, B_DMD = operator
    w_est = emulate_DMD(F, A_DMD, B_DMD, w0, n_steps)
  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = operator
    w_est = emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F)
  elif op_type == 'deconvolve' or op_type == 'direct' or op_type == 'FDT' or op_type == 'fit' or op_type == 'fit_DMD' or op_type == 'fit_jax' or op_type == 'fit_amp' or op_type == 'fit_complex':
    w_est = emulate_response(F, operator, dt)
  elif op_type == 'PS':
    pattern = operator
    w_est = emulate_PS(w, pattern)
  elif op_type == 'QE':
    w_est = QE_interp(operator, F, n_boxes)
  else:
    raise ValueError(f'Operator type {op_type} not recognized.')

  return w_est


#####################
## Error Functions ##
#####################
def calc_NRMSE(w_true, w_est):
  return calc_RMSE(w_true, w_est)/np.abs(np.mean(w_true,axis=1))*100

def calc_RMSE(w_true, w_est):
  return np.sqrt(np.mean((w_true - w_est)**2,axis=1))

def calc_MAE(w_true, w_est):
  return np.mean(np.abs(w_true - w_est),axis=1)

def calc_Bias(w_true, w_est):
  return np.mean(w_true - w_est,axis=1)

def calc_MRE(w_true, w_est):
  return 100*np.mean(np.divide(w_true - w_est, w_true),axis=1)

def calc_L2(w_true, w_est):
  # Estimate L2 error between emulator and ground truth
  return np.linalg.norm(w_true - w_est)

def calc_error_metrics(w_true, w_est):
  #return [calc_RMSE(w_true, w_est), calc_MAE(w_true, w_est), calc_Bias(w_true, w_est), calc_MRE(w_true, w_est)]
  return calc_NRMSE(w_true, w_est)

def emulate_scenarios(op_type, scenarios=None, outputs=None, forcings=None, w0=None, t=None, dt=None, n_steps=None, n_boxes=None,
                        w_dict=None, F_dict=None, n_modes=None, verbose=True, diff_flag=0, vert_diff_flag=0, B=None, xi=0, n_ensemble=None,
                        delta=0, t_range=None, regularize=False, spatial_flag=0, exp=None):
  operator, w_pred, error_metrics = {}, {}, {}

  if op_type == 'direct':
    operator = create_emulator(op_type, None, None, n_boxes=n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, spatial_flag=spatial_flag, delta=delta)
    if verbose:
      print(f'Train: Impulse Forcing - L2 Error')

    for scen2 in scenarios:
      F2 = forcings[scen2]
      w_true_2 = outputs[scen2]
      w_pred[scen2] = estimate_w(F2, operator, op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
      error_metrics[scen2] = calc_error_metrics(w_true_2, w_pred[scen2])
      #print(f'\tTest: {scen2} - {L2[scen2]}')

  else:
    for scen1 in scenarios:
      w_pred[scen1], error_metrics[scen1] = {}, {}
      if verbose:
        print(f'Train: {scen1} - L2 Error')

      F1, w_true_1 = forcings[scen1], outputs[scen1]
      F1, w_true_1 = check_dim(F1), check_dim(w_true_1)

      # Crop a specific range of time for training
      if t_range is not None:
        F1, w_true_1 = F1[:,t_range], w_true_1[:,t_range]

      operator[scen1] = create_emulator(op_type, w_true_1, F1, t=t, dt=dt, n_boxes=n_boxes, w_dict=w_dict,
                                       F_dict=F_dict, n_modes=n_modes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag,
                                       B=B, xi=xi, n_ensemble=n_ensemble, delta=delta, n_steps=n_steps, regularize=regularize,
                                       verbose=verbose, exp=exp)

      for scen2 in scenarios:
        F2, w_true_2 = forcings[scen2], outputs[scen2]
        F2, w_true_2 = check_dim(F2), check_dim(w_true_2)
        w_pred[scen1][scen2] = estimate_w(F2, operator[scen1], op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict, w=w_true_2)
        error_metrics[scen1][scen2] = calc_error_metrics(w_true_2, w_pred[scen1][scen2])
        if verbose:
          print(f'\tTest: {scen2} - {error_metrics[scen1][scen2]}')

  return operator, w_pred, error_metrics

###############################
## Ensemble Helper Functions ##
###############################

def evaluate_ensemble(scenarios, n_ensemble, n_choices, forcings_ensemble, w_ensemble, op_type, op_true, w0=None, n_steps=None,
                      t=None, dt=None, n_boxes=None, w_dict=None, F_dict=None, n_modes=None, diff_flag=0, vert_diff_flag=0, B=None, xi=0, delta=0, regularize=False):

  operator_true, operator_avg, operator_L2_avg, w_pred_L2 = {}, {}, {}, {}

  # Separate treament for direct diagnosis of response function (does this even make sense in this context?)
  if op_type == 'direct':
    return False
    """
    operator_ensemble, operator_L2_avg = [], []

    # Generate noisy ensemble for direct experiment
    for n in range(n_ensemble):
      operator_ensemble.append(method_2a_direct(n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi))

    # Iterate over ensemble subsets of length n
    for n in range(1,n_ensemble + 1):
      operator_subset, operator_L2_subset = [], []

      for i in range(n_choices):
        operator_temp = random.sample(operator_ensemble, n)
        operator_temp_mean = np.mean(np.stack(operator_temp, axis=0), axis=0)
        operator_subset.append(operator_temp_mean)

        # Calculate error between ensemble and ground-truth operator
        operator_L2_subset.append(np.linalg.norm(np.array(operator_temp_mean) - np.array(op_true)))

      # Calculate the average operator and error across the number of choices
      R_mean = np.mean(np.stack(operator_subset, axis=0), axis=0)
      operator_ensemble.append((R_mean))
      operator_L2_avg.append(np.mean(operator_L2_subset))

    return operator_ensemble, operator_L2_avg
    """

  for scen_flag, scen1 in enumerate(scenarios):
    forcing_w = list(zip(forcings_ensemble[scen1], w_ensemble[scen1]))
    operator_avg[scen1], operator_L2_avg[scen1], w_pred_L2[scen1] = [], [], {}

    # Take mean over the entire ensemble
    mean_forcing = np.mean(np.stack(forcings_ensemble[scen1], axis=0), axis=0)
    mean_w = np.mean(np.stack(w_ensemble[scen1], axis=0), axis=0)

    operator_true[scen1] = create_emulator(op_type, mean_w, mean_forcing, t=t, dt=dt, n_boxes=n_boxes, w_dict=w_dict,
                                       F_dict=F_dict, n_modes=n_modes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag,
                                       B=B, xi=xi, n_ensemble=n_ensemble, delta=delta, n_steps=n_steps, regularize=regularize)

    # Setup data storage
    for scen2 in scenarios:
      w_pred_L2[scen1][scen2] = []

    # Iterate over ensemble subsets of length n
    for n in range(1,n_ensemble + 1):
      operator_subset, operator_L2_subset, w_pred_L2_subset = [], [], {}

      for scen2 in scenarios:
        w_pred_L2_subset[scen2] = []

      # Repeatedly select subsets n_choices times
      for _ in range(n_choices):
        temp_choice = random.sample(forcing_w, n)
        forcing_choice, w_choice = zip(*temp_choice)

        # Take mean over the subset of ensemble members
        mean_forcing_subset = np.mean(np.stack(forcing_choice, axis=0), axis=0)
        mean_w_subset = np.mean(np.stack(w_choice, axis=0), axis=0)

        # Calculate operator over the subset
        if op_type == 'DMD':
          A_DMD, B_DMD = method_1a_DMD(mean_w_subset, mean_forcing_subset)
          operator_temp = (A_DMD, B_DMD)
        elif op_type == 'EDMD':
          A_EDMD, B_EDMD = method_1b_EDMD(mean_w_subset, mean_forcing_subset, w_dict, F_dict)
          operator_temp = (A_EDMD, B_EDMD)
        elif op_type == 'deconvolve':
          operator_temp = method_3a_deconvolve(mean_w_subset, mean_forcing_subset, dt, regularize=True)
        elif op_type == 'fit':
          operator_temp = method_4a_fit(mean_w_subset, mean_forcing_subset, t, dt, n_modes, n_boxes, B)
        elif op_type == 'FDT':
          operator_temp = method_2b_FDT(n, n_boxes, n_steps, xi, delta, scen_flag, diff_flag, vert_diff_flag)

        operator_subset.append(operator_temp)

        # Calculate error between ensemble and ground-truth operator
        operator_L2_subset.append(calc_NRMSE(np.array(operator_temp), np.array(op_true[scen1])))

        # Emulate output and calculate L2 to ground truth
        for scen2 in scenarios:
          forcing_true = np.mean(forcings_ensemble[scen2], axis=0)
          w_true = np.mean(w_ensemble[scen2], axis=0)

          w_pred_temp = estimate_w(forcing_true, operator_temp, op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
          w_pred_L2_subset[scen2].append(calc_NRMSE(w_true, w_pred_temp))

      # Calculate the average operator and error across the number of choices
      if op_type == 'DMD' or op_type == 'EDMD':
        A, B = zip(*operator_subset)
        A_mean, B_mean = np.mean(np.stack(A, axis=0), axis=0), np.mean(np.stack(B, axis=0), axis=0)
        operator_avg[scen1].append((A_mean, B_mean))
      elif op_type == 'deconvolve' or op_type == 'fit':
        R_mean = np.mean(np.stack(operator_subset, axis=0), axis=0)
        operator_avg[scen1].append((R_mean))

      operator_L2_avg[scen1].append(np.mean(operator_L2_subset))

      for scen2 in scenarios:
        w_pred_L2[scen1][scen2].append(np.mean(w_pred_L2_subset[scen2]))

  return operator_true, operator_avg, operator_L2_avg, w_pred_L2

##############################
## General Helper Functions ##
##############################
def save_error(metrics, name):
  with open(f'Error Results/{name}.pkl', 'wb') as file:
    pickle.dump(metrics, file)
  return

def open_error(name):
  with open(f'Error Results/{name}.pkl', 'rb') as file:
    metric = pickle.load(file)
  return metric

def get_regularization(w, F):
  init_params = np.random.rand(2)  # Start with log(1), log(1)

  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  res = minimize(fit_opt_hyper,
                 init_params,
                 args=(w, F_toep),
                 method='L-BFGS-B')

  return res.x

def fit_opt_hyper(params, w, F_toep):
  sig2, lam2 = params

  n_x, n_t = w.shape
  Sig = sig2*np.eye(n_t) + lam2*(F_toep @ F_toep.T)

  sign, logdet_Sig = np.linalg.slogdet(Sig)
  if sign <= 0:
    return np.inf  # Avoid invalid log-determinant

  Sig_inv_w = np.linalg.solve(Sig, w.T)
  quadratic_term = np.sum(w * Sig_inv_w.T)
  log_evidence_value = -0.5 * (n_x * logdet_Sig + quadratic_term)

  return -log_evidence_value  # Negate for minimization

def plot_true_pred(T_true, T_pred, experiments, operator=None):
  n_exp = len(experiments)
  fig, ax = plt.subplots(n_exp, n_exp, figsize=(12,12), constrained_layout=True, sharex='col', sharey='row')

  for i, exp1 in enumerate(experiments):
    for j, exp2 in enumerate(experiments):
      T_true_temp, T_pred_temp = T_true[exp2].T, T_pred[exp1][exp2].T
      T_true_temp, T_pred_temp = check_dim(T_true_temp, transp=True), check_dim(T_pred_temp, transp=True)

      n_boxes = T_true_temp.shape[1]

      if operator == 'PS':
        ax[i,j].plot(np.mean(T_true_temp,axis=1), lw=2, c='k', alpha=0.75)

      for k in range(n_boxes):
        ax[i,j].plot(T_true_temp[:,k], lw=2, c=brewer2_light(k), alpha=0.5)
        ax[i,j].plot(T_pred_temp[:,k], lw=2, c=brewer2_light(k), ls='-.')

  # Add experiment names as column titles (train)
  for j, train_name in enumerate(experiments):
    ax[n_exp-1, j].set_xlabel(train_name, fontsize=14, labelpad=10)

  # Add experiment names as row labels (test)
  for i, test_name in enumerate(experiments):
    ax[i, 0].set_ylabel(test_name, fontsize=14, rotation=90, labelpad=10, va="center")

  # Label overall X and Y axes
  fig.text(0.5, -0.02, "Train Scenario", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(-0.02, 0.5, "Test Scenario", ha="center", va="center", fontsize=18, fontweight="bold", rotation=90)

  return

def plot_true_pred_direct(T_true, T_pred, experiments):
  n_exp = len(experiments)
  fig, ax = plt.subplots(1, n_exp, figsize=(12,4), constrained_layout=True)

  for i, exp in enumerate(experiments):
    T_true_temp, T_pred_temp = T_true[exp].T, T_pred[exp].T
    n_boxes = T_true_temp.shape[1]
    for k in range(n_boxes):
      ax[i].plot(T_true_temp[:,k], lw=2, c=brewer2_light(k), alpha=0.5)
      ax[i].plot(T_pred_temp[:,k], lw=2, c=brewer2_light(k), ls='-.')

  # Add experiment names as row labels (test)
  for i, test_name in enumerate(experiments):
    ax[i].set_xlabel(test_name, fontsize=14, labelpad=10, va="center")

  # Label overall X and Y axes
  fig.text(0.5, -0.02, "Test Scenario", ha="center", va="center", fontsize=18, fontweight="bold")

  fig.suptitle("Method X: Direct Diagnosis")

  return

#######################################
## Dictionary Basis Function Classes ##
#######################################

class Vector_Dict:
  def __init__(self, method="polynomial", **kwargs):
    """
    Initialize the dictionary with a chosen method.

    Available methods:
    - 'polynomial': Polynomial basis up to 'degree'
    - 'rbf': Radial basis functions with 'centers' and 'gamma'
    - 'hermite' : Hermite polynomial expansions

    kwargs can include:
    - degree (int): polynomial/Hermite degree
    - centers (array), gamma (float): for RBF
    """
    self.method = method
    self.params = kwargs

  def transform(self, X):
    """
    Lift input X into a higher-dimensional space.
    X should be shape (N_samples, N_features).
    Returns shape (N_samples, N_lifted_features).
    """
    if self.method == "polynomial":
      return self._poly_dictionary(X, self.params.get("degree", 2))
    elif self.method == "poly_cross":
      return self._poly_cross_dictionary(X, self.params.get("degree", 2))
    elif self.method == "rbf":
      return self._rbf_dictionary(X,
                                  self.params.get("centers", None),
                                  self.params.get("gamma", 1.0))
    elif self.method == "hermite":
      return self._hermite_dictionary(X, self.params.get("degree", 2))
    elif self.method == "damped_osc":
      return self._damped_osc_dictionary(
        X,
        alpha=self.params.get("alpha", 1.0),
        omega=self.params.get("omega", 1.0)
        )
    elif self.method == "deriv":
      return self._data_and_derivative_dictionary(X)
    else:
      raise ValueError(f"Unknown dictionary method: {self.method}")

  def _poly_dictionary(self, X, degree=2):
    """Simple polynomial features up to 'degree' (no cross-terms)."""
    N, D = X.shape
    # Start with constant term
    Phi = [np.ones(N)]
    # Add powers of each feature up to 'degree'
    for d in range(1, degree+1):
      for i in range(D):
        Phi.append(X[:, i]**d)
    return np.vstack(Phi).T

  def _poly_cross_dictionary(self, X, degree=2):
    """
    Polynomial features including cross terms up to 'degree' = 2.
    e.g. for D=3, we get:
      [1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3].
    Extend similarly if you need higher degrees.
    """
    N, D = X.shape
    Phi = [np.ones(N)]  # constant term

    if degree >= 1:
      # linear terms: x1, x2, ...
      for i in range(D):
        Phi.append(X[:, i])
    if degree >= 2:
      # squares: x1^2, x2^2, ...
      for i in range(D):
        Phi.append(X[:, i]**2)
      # cross terms: x1*x2, x1*x3, etc.
      for i in range(D):
        for j in range(i+1, D):
          Phi.append(X[:, i] * X[:, j])

    return np.vstack(Phi).T

  def _rbf_dictionary(self, X, centers=None, gamma=1.0):
    """Radial Basis Function dictionary with optional 'centers'."""
    N, D = X.shape
    if centers is None:
      # Create 5 equally spaced centers per dimension
      # (for illustration; adjust as needed)
      centers = []
      for dim in range(D):
        c_lin = np.linspace(np.min(X[:,dim]), np.max(X[:,dim]), 5)
        centers.append(c_lin)
      # Create all combinations of centers across dimensions
      from itertools import product
      centers = np.array(list(product(*centers)))
    # Build RBF features
    Phi = [np.ones(N)]
    for center in centers:
      dist_sq = np.sum((X - center)**2, axis=1)
      Phi.append(np.exp(-gamma * dist_sq))
    return np.vstack(Phi).T

  def _hermite_dictionary(self, X, degree=2):
    """
    For each feature x_i, compute Hermite polynomials H_d(x_i)
    for d=0,...,degree. Then stack them all (no cross terms).
    H_0(x)=1 acts like a constant term for each feature dimension.

    The final shape will be (N_samples, (degree+1)*D).
    """
    N, D = X.shape
    # We'll accumulate each basis as a row in 'Phi_list'
    Phi_list = []
    for d in range(1,degree + 1):
      if d == 1:
        # First, prepend non-lifted components
        for i in range(D):
          Phi_list.append(X[:, i])

      # Now apply the Hermite polynomials
      for i in range(D):
        # Vectorize eval_hermite(d, x)
        herm_vals = eval_hermite(d, X[:, i])
        Phi_list.append(herm_vals)
        # Stack all results => shape (n_basis, N) => transpose => (N, n_basis)
    return np.vstack(Phi_list).T

  def _damped_osc_dictionary(self, X, alpha=1.0, omega=1.0):
    """
    Builds a dictionary for each dimension that includes:
      1. e^{-alpha * x_i}          (pure exponential decay)
      2. e^{-alpha * x_i} cos(omega * x_i)
      3. e^{-alpha * x_i} sin(omega * x_i)

    If X has shape (N_samples, D), each dimension i is expanded
    into these 3 columns. We also include a global constant term.

    Output shape: (N_samples, 1 + 3*D)
      -> 1 is for the constant,
          3*D is for each dimension's damped oscillator expansions.
    """
    N, D = X.shape
    # Start with a global constant
    Phi = []

    # First, prepend non-lifted components
    for i in range(D):
      Phi.append(X[:, i])

    for i in range(D):
      xi = X[:, i]
      exp_term = np.exp(-alpha * xi)
      Phi.append(exp_term)                         # e^{-alpha x_i}
      Phi.append(exp_term * np.cos(omega * xi))    # e^{-alpha x_i} cos(omega x_i)
      Phi.append(exp_term * np.sin(omega * xi))    # e^{-alpha x_i} sin(omega x_i)

    return np.vstack(Phi).T

  def _data_and_derivative_dictionary(self, X):
    """
    Creates a dictionary from the input data and its discrete derivative.

    The derivative is approximated using finite differences, assuming the
    samples in X are sequential and equally spaced in time. It uses a
    forward difference for all points except the last, for which a
    backward difference is used.

    If X has shape (N_samples, N_features), the output dictionary will
    have shape (N_samples, 2 * N_features).
    """
    N, D = X.shape

    # Handle cases with insufficient data to compute a derivative
    if N < 2:
      # Return the data concatenated with columns of zeros for the derivative
      return np.hstack([X, np.zeros_like(X)])

    # Initialize an array to hold the derivative approximation
    X_dot = np.zeros_like(X)

    # Compute derivative using forward difference for all but the last point
    # X_dot[t] = X[t+1] - X[t]
    X_dot[:-1, :] = np.diff(X, axis=0)

    # For the last point, use a backward difference
    # X_dot[-1] = X[-1] - X[-2]
    X_dot[-1, :] = X[-1, :] - X[-2, :]

    # Concatenate the original data and its derivative side-by-side
    # The new features are [x_1, ..., x_D, x_dot_1, ..., x_dot_D]
    return np.hstack([X, X_dot])


###################
## Plot Colormap ##
###################
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