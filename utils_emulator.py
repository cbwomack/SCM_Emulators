# Required imports
## Basic imports
import numpy as np
import random
import pickle
import utils_BudykoSellers

## Math
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import solve_ivp

## Optimization
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from jax.example_libraries import optimizers

## Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cmcrameri import cm

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

#####################################
## Functions to Diagnose Emulators ##
#####################################

def method_Ia_PS(w):
  """
  Derive pattern-scaling coefficients from spatial data.

  Parameters
  ----------
  w : ndarray
    Array of shape (n_space, n_time) containing spatial values over time.

  Returns
  -------
  pattern : ndarray
    Row vector (1, n_space) of pattern-scaling coefficients.
  """

  # Take global mean
  nx = len(w)
  global_mean = np.mean(w,axis=0).reshape(-1, 1)

  # Perform least squares to find pattern
  XTX_inv = np.linalg.inv(global_mean.T @ global_mean)
  XTY = global_mean.T @ w.T
  pattern = (XTX_inv @ XTY).reshape(1, nx)

  return pattern

def method_Ib_PS_forcing(w, F):
  """
  Estimate spatial pattern coefficients using an external forcing.

  Parameters
  ----------
  w : ndarray
    Spatial data, shape (n_space, n_time) or (n_time,).
  F : ndarray
    Forcing time series, shape (n_time,).

  Returns
  -------
  pattern : ndarray
    Row vector (1, n_space) of pattern coefficients.
  """

  # Ensure both inputs are arrays and reshape to (1, n_time) where needed
  F = np.asarray(F)
  F = F.reshape(1, -1)
  w = np.asarray(w)

  # Confirm matching time dimensions
  if w.ndim == 1:
    w = w.reshape(1, -1)
  elif w.shape[0] == F.shape[1] and w.shape[1] != F.shape[1]:
    w = w.T

  if w.shape[1] != F.shape[1]:
    raise ValueError("Time dimension of w must match that of F")

  # Perform least squares to find pattern
  XtX = F @ F.T
  pattern = (1.0 / XtX) @ (F @ w.T)

  return pattern

def method_IIa_FDT(n_boxes, diff_flag=0, vert_diff_flag=0, xi=0, spatial_flag=0, delta=0):
  """
  Compute the linear temperature-response matrix G_FDT using
  a finite-difference approximation of the Fluctuation-Dissipation
  Theorem (FDT) with the Budyko-Sellers model.

  Parameters
  ----------
  n_boxes : int
    Number of spatial boxes (latitude bands) to retain.
  diff_flag, vert_diff_flag, xi, spatial_flag : int or float, optional
    Model configuration flags passed through to
    `utils_BudykoSellers.Run_Budyko_Sellers`.
  delta : float, optional
    Small perturbation applied to the forcing (default 0).

  Returns
  -------
  G_FDT : ndarray
    Array of shape (n_boxes, n_time) giving the linear response
    of temperature to the perturbation.
  """

  # Run unperturbed and perturbed simulations
  full_output_unpert = utils_BudykoSellers.Run_Budyko_Sellers(scen_flag=4, diff_flag=diff_flag,
                                                              vert_diff_flag=vert_diff_flag, xi=xi,
                                                              spatial_flag=spatial_flag)

  full_output_pert = utils_BudykoSellers.Run_Budyko_Sellers(scen_flag=4, diff_flag=diff_flag,
                                                            vert_diff_flag=vert_diff_flag, xi=xi,
                                                            spatial_flag=spatial_flag, delta=delta)

  # Approximate of the response matrix
  G_FDT = (np.squeeze(full_output_pert['T_ts'])[0:n_boxes,:] - np.squeeze(full_output_unpert['T_ts'])[0:n_boxes,:])/delta

  return G_FDT

def method_IIb_FDT(n_ensemble, n_boxes, n_steps, xi, delta, scen_flag=0, diff_flag=0, vert_diff_flag=0):
  """
  Ensemble-based FDT estimate of the linear temperature-response matrix.

  Parameters
  ----------
  n_ensemble : int
    Number of ensemble members.
  n_boxes : int
    Spatial boxes (latitude bands) to retain.
  n_steps : int
    Time steps per simulation.
  xi, delta : float
    Model parameter and perturbation magnitude.
  scen_flag, diff_flag, vert_diff_flag : int, optional
    Scenario and model configuration flags.

  Returns
  -------
  G_FDT : ndarray
    (n_boxes, n_steps) ensemble-mean response matrix.
  """

  # Allocate arrays for unperturbed and perturbed outputs
  w, w_delta = np.zeros((n_ensemble, n_boxes, n_steps)), np.zeros((n_ensemble, n_boxes, n_steps))

  # Run ensemble: first unperturbed, then perturbed with identical noise
  for n in range(n_ensemble):
    # Unperturbed simulation
    full_output_unperturbed = utils_BudykoSellers.Run_Budyko_Sellers(scen_flag=scen_flag, diff_flag=diff_flag,
                                                                     vert_diff_flag=vert_diff_flag, xi=xi)
    w[n,:,:] = np.squeeze(full_output_unperturbed['T_ts'])[0:n_boxes,:]
    noise_ts = full_output_unperturbed['noise_ts']

    # Perturbed simulation using same stochastic noise
    full_output_perturbed = utils_BudykoSellers.Run_Budyko_Sellers(scen_flag=scen_flag, diff_flag=diff_flag,
                                                                   vert_diff_flag=vert_diff_flag, xi=xi, delta=delta,
                                                                   noise_ts=noise_ts)
    w_delta[n,:,:] = np.squeeze(full_output_perturbed['T_ts'])[0:n_boxes,:]

  # Approximate of the response matrix
  G_FDT = np.mean(w_delta - w, axis=0)/delta

  return G_FDT

def method_III_deconvolve(w, F, dt, regularize=False):
  """
  Recover the linear kernel G by deconvolving a forcing time
  series from the system response.

  Parameters
  ----------
  w : ndarray
    Spatial data (n_space, n_time).
  F : ndarray
    Forcing series, shape (n_time,) or (1, n_time).
  dt : float
    Time step used for scaling.
  regularize : bool, optional
    If True, apply ridge regularization (default False).

  Returns
  -------
  G_deconv : ndarray
    Estimated kernel, shape (n_space, n_time), scaled by 1/dt.
  """
  # Ensure forcing has shape (1, n_time)
  F = check_dim(F)

  # Toeplitz matrix for the convolution operator
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  if regularize:
    # Ridge penalty based on noise / signal variance
    sig2, lam2 = get_regularization(w, F)
    alpha = sig2 / lam2
    G_deconv = np.linalg.solve(F_toep.T @ F_toep + alpha * np.eye(F.shape[1]), F_toep.T @ w.T)

  else:
    # Direct triangular solve (no regularization)
    G_deconv = spsolve_triangular(F_toep, w.T, lower=True)

  return G_deconv.T/dt

def method_IVa_modal(w, F, t, dt, n_modes, n_boxes,
                     num_steps=1000, verbose=False):
  """
  Fit a modal (eigen-mode) representation of the linear kernel G
  by optimising eigen-amplitudes and decay rates with JAX/ADAM.

  Parameters
  ----------
  w : ndarray
    Spatial data, shape (n_boxes, n_time).
  F : ndarray
    Forcing series, shape (1, n_time) or (n_time,).
  t : ndarray
    Time vector (n_time,).
  dt : float
    Time step.
  n_modes : int
    Number of exponential modes to fit.
  n_boxes : int
    Spatial boxes retained.
  num_steps : int, optional
    Optimiser iterations (default 1000).
  verbose : bool, optional
    If True, print final optimiser parameters.

  Returns
  -------
  G_modal : ndarray
    Fitted kernel, shape (n_boxes, n_time).
  """

  # Set up tensors
  F_toep = sparse.csr_matrix(toeplitz(F[0, :], np.zeros_like(F[0, :])))
  F_toep = jnp.asarray(F_toep.toarray())
  dt     = jnp.asarray(dt)
  w      = jnp.asarray(w)
  t      = jnp.asarray(t)

  # Build JIT-compiled cost function (L2 error)
  def make_fit_opt_eigs_jax(n_modes: int, n_boxes: int):

    @jax.jit
    def fit_opt_eigs_amp(params, w, F_toep, t, dt):
      # Unpack parameters into amplitudes (beta) and decay rates (lam)
      beta   = params[:n_boxes*n_modes].reshape((n_boxes, n_modes))
      theta  = params[n_boxes*n_modes:].reshape((n_modes, 1))
      lam    = -jnp.exp(theta)

      # Enforce sign/structure constraints
      alpha  = jnp.exp(beta)
      alpha  = alpha.at[jnp.diag_indices(min(n_boxes, n_modes))]\
                    .set(1.0 / jnp.exp(jnp.diagonal(beta)))

      # Kernel construction and model prediction (0th entry of discrete G is zero)
      exp_lam_t = jnp.exp(lam * t)
      G_opt = alpha @ exp_lam_t
      G_opt = G_opt.at[:, 0].set(0.0)
      model = (G_opt * dt) @ F_toep.T

      # L1 penalty to promote sparsity / block-diagonal alpha
      l1_penalty = 100.0 * jnp.sum(
        jnp.abs(alpha * (1.0 - jnp.eye(alpha.shape[0])))
      )
      return jnp.linalg.norm(w - model, ord=2) + l1_penalty

    return fit_opt_eigs_amp

  # Initial guess for amplitudes and eigenvalues, experiment-specific
  if n_modes == 3:
      initial_phi   = np.log([1e2, 1e-3, 1e-3,
                              1e-3, 1,    1e-3,
                              1e-3, 1e-3, 1e1])
      initial_theta = np.log([1e-3, 1e-1, 1e-2])
  elif n_boxes == 1 and n_modes == 2:
      initial_phi   = np.log([1e2, 1e-3])
      initial_theta = np.log([1e-3, 1e-1])
  initial_params = jnp.concatenate([initial_phi, initial_theta])

  # Optimize using ADAM
  fit_fn   = make_fit_opt_eigs_jax(n_modes, n_boxes)

  @jax.jit
  def cost_fn(params, w, F_toep, t, dt):
    return fit_fn(params, w, F_toep, t, dt)

  value_and_grad = jax.jit(jax.value_and_grad(cost_fn))
  learning_rate  = 0.1
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)
  opt_state = opt_init(initial_params)

  @jax.jit
  def update(step, opt_state, w, F_toep, t, dt):
    params = get_params(opt_state)
    val, grads = value_and_grad(params, w, F_toep, t, dt)
    return opt_update(step, grads, opt_state)

  for step_i in range(num_steps):
    opt_state = update(step_i, opt_state, w, F_toep, t, dt)

  opt_params = get_params(opt_state)

  if verbose:
    print(opt_params)

  # Construct and return response function
  return apply_modal(opt_params, t, n_modes, n_boxes)

def apply_modal(params, t, n_modes, n_boxes):
  """
  Build a modal response function G(t) from fitted parameters.

  Parameters
  ----------
  params : ndarray
    Flattened amplitudes and decay rates from optimisation.
  t : ndarray
    Time vector.
  n_modes : int
    Number of exponential modes.
  n_boxes : int
    Spatial boxes retained.

  Returns
  -------
  G_opt : ndarray
    Modal response function, shape (n_boxes, len(t)).
  """

  # Unpack parameters into amplitudes (beta) and decay rates (lam)
  beta = params[:n_boxes*n_modes].reshape((n_boxes, n_modes))
  theta = params[n_boxes*n_modes:].reshape((n_modes, 1))
  lam = -jnp.exp(theta)

  # Amplitude matrix with enforced inverse diagonal structure
  alpha = jnp.exp(beta)
  alpha_diag = 1.0 / jnp.exp(jnp.diagonal(beta))
  alpha = alpha.at[jnp.diag_indices(n_boxes)].set(alpha_diag)

  # Assemble response function G_opt over time
  G_opt = np.zeros((n_boxes, len(t)))
  for i, n in enumerate(t):
    # Ensure G(0) = 0
    if i == 0:
      continue
    G_opt[:,i] = (alpha @ jnp.exp(lam*n)).reshape(n_boxes)

  return G_opt

def method_IVb_modal_complex(w, F, t, dt, n_modes, n_boxes, num_steps=1000, learning_rate=0.1, verbose=False):
  """
  Fit a complex-valued modal response function with oscillatory
  components via ADAM optimisation in JAX.

  Parameters
  ----------
  w : ndarray
    Spatial data, shape (n_boxes, n_time).
  F : ndarray
    Forcing series, shape (1, n_time) or (n_time,).
  t : ndarray
    Time vector (n_time,).
  dt : float
    Time step.
  n_modes : int
    Number of complex modes to fit.
  n_boxes : int
    Spatial boxes retained.
  num_steps : int, optional
    Optimiser iterations (default 1000).
  learning_rate : float, optional
    ADAM learning rate (default 0.1).
  verbose : bool, optional
    If True, print cost every step and final parameters.

  Returns
  -------
  response_func : ndarray
    Complex response function, shape (n_boxes, n_time).
  """

  # Check shapes and transfer arrays to device
  F = check_dim(F)
  w = check_dim(w)
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  F_toep_T = jnp.array(F_toep.T.toarray())
  dt = jnp.array(dt)
  w = jnp.array(w)
  t = jnp.array(t)

  # Build cost function (L2 error)
  def cost_fn(params, w_target, t_vec, F_matrix_T):
    phi = params[:n_boxes]
    theta = params[n_boxes:2*n_boxes]
    beta = params[2*n_boxes:]

    alpha = jnp.exp(phi)
    lam = -jnp.exp(theta)
    omega = jnp.exp(beta)

    alpha_r = alpha[:, jnp.newaxis]
    lam_r = lam[:, jnp.newaxis]
    omega_r = omega[:, jnp.newaxis]
    t_r = t_vec[jnp.newaxis, :]

    G_opt = (1/alpha_r) * jnp.exp((1/alpha_r) * (lam_r + 1j*omega_r) * t_r)

    model = (G_opt * dt) @ F_matrix_T
    return jnp.linalg.norm(w_target - model, ord=2)

  # Optimizer setup
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)
  key, phi_key, theta_key, beta_key = jax.random.split(jax.random.PRNGKey(), 4)
  initial_phi = jax.random.normal(phi_key, (n_modes,))
  initial_theta = jax.random.normal(theta_key, (n_modes,))
  initial_beta = jax.random.normal(beta_key, (n_modes,))
  initial_params = jnp.concatenate([initial_phi, initial_theta, initial_beta])
  opt_state = opt_init(initial_params)

  @jax.jit
  def update_step(step, opt_state, w_target, t_vec, F_matrix_T):
    params = get_params(opt_state)
    loss_value, grads = jax.value_and_grad(cost_fn)(params, w_target, t_vec, F_matrix_T)
    new_opt_state = opt_update(step, grads, opt_state)
    return new_opt_state, loss_value

  # Optimization loop
  for step_i in range(num_steps):
    opt_state, cval = update_step(step_i, opt_state, w, t, F_toep_T)

    if verbose:
      print(f"Step {step_i}, cost={cval:0.6f}")

  opt_params = get_params(opt_state)

  if verbose:
    print("Final optimized parameters:", opt_params)

  # Build and return complex response function
  return apply_complex(opt_params, t, n_boxes)


def apply_complex(params, t, n_boxes):
  """
  Construct a complex-valued modal response function from
  optimised parameters.

  Parameters
  ----------
  params : ndarray
    Flattened array of log-amplitudes, log-decay rates, and
    log-frequencies (length 3 x n_boxes).
  t : ndarray
    Time vector.
  n_boxes : int
    Number of spatial boxes retained.

  Returns
  -------
  ndarray
    Complex response function of shape (n_boxes, len(t)).
  """

  # Split flattened parameter vector
  phi = params[:n_boxes]
  theta = params[n_boxes:2*n_boxes]
  beta = params[2*n_boxes:]

  # Convert log-parameters to physical values
  alpha = jnp.exp(phi)
  lam = -jnp.exp(theta)
  omega = jnp.exp(beta)

  # Allocate output array
  G_opt = np.zeros((n_boxes, len(t)))

  # Assemble response function G_opt over time
  for i, n in enumerate(t):
    # Ensure G(0) = 0
    if i == 0:
      continue
    G_opt[:,i] = 1/alpha*jnp.exp(1/alpha*(lam + 1j*omega)*n)

  return G_opt

def method_V_DMD(w, F, regularize=False, lam=1e-4):
  """
  Dynamic Mode Decomposition with control to estimate the
  linear state matrix A_DMD and control matrix B_DMD.

  Parameters
  ----------
  w : ndarray
    State snapshots, shape (n_space, n_time).
  F : ndarray
    Forcing snapshots, shape (n_space, n_time).
  regularize : bool, optional
    If True, apply ridge regularisation (default False).
  lam : float, optional
    Ridge penalty parameter (default 1e-4).

  Returns
  -------
  A_DMD : ndarray
    Estimated state-transition matrix, shape (n_space, n_space).
  B_DMD : ndarray
    Estimated control-response matrix, shape (n_space, n_space).
  """
  # Reshape inputs and build augmented data matrix
  w, F = check_dim(w), check_dim(F)
  Omega = np.concatenate([w[:,:-1],F[:,:-1]])

  # Optionally apply ridge regularization
  if regularize:
    G = Omega @ Omega.T
    reg = lam * np.eye(G.shape[0])
    L = w[:,1:] @ Omega.T @ np.linalg.inv(G + reg)
  else:
    L = w[:,1:] @ np.linalg.pinv(Omega)

  # Split combined operator into A and B blocks
  n = len(w)
  A_DMD, B_DMD = L[:,:n], L[:,n:]

  return A_DMD, B_DMD


def method_VI_EDMD(w, F, dict_w, dict_F, regularize=False, lam=1e-3):
  """
  Extended Dynamic Mode Decomposition with control.

  Parameters
  ----------
  w : ndarray
    State snapshots, shape (n_space, n_time).
  F : ndarray
    Forcing snapshots, shape (n_space, n_time).
  dict_w, dict_F : sklearn-like transformer
    Feature dictionaries for states and forcings.
  regularize : bool, optional
    If True, apply ridge regularisation (default False).
  lam : float, optional
    Ridge penalty parameter.

  Returns
  -------
  A_EDMD : ndarray
    State-state transition matrix in feature space.
  B_EDMD : ndarray
    Forcing-state response matrix in feature space.
  """
  # Standardize array orientation
  w, F = check_dim(w), check_dim(F)
  F = F.T

  # Transform data into feature space
  Phi_F = dict_F.transform(F[:-1,:])
  Phi_w = dict_w.transform(w[:,:-1].T)
  Phi_wprime = dict_w.transform(w[:,1:].T)

  # Assemble augmented matrix
  Omega = np.concatenate([Phi_w.T,Phi_F.T])

  # Optionally apply ridge regularization
  if regularize:
    G = Omega @ Omega.T
    reg = lam * np.eye(G.shape[0])
    K = Phi_wprime.T @ Omega.T @ np.linalg.inv(G + reg)
  else:
    K = Phi_wprime.T @ np.linalg.pinv(Omega)

  # Split combined operator into A and B blocks
  n = len(Phi_wprime.T)
  A_EDMD, B_EDMD = K[:,:n], K[:,n:]

  return A_EDMD, B_EDMD

def create_emulator(op_type, w, F, t=None, dt=None, n_boxes=None, w_dict=None, F_dict=None, n_modes=None, diff_flag=0,
                    vert_diff_flag=0, B=None, xi=0, n_ensemble=None, n_steps=None, delta=0, regularize=False, verbose=False,
                    spatial_flag=0, exp=None):
  """
  Wrapper that builds an emulator operator of the requested
  type from training data and settings.

  Parameters
  ----------
  op_type : str
    Choice of operator {'PS', 'PS_forcing', 'FDT', 'deconvolve',
    'modal', 'fit_complex', 'DMD', 'EDMD'}.
  w, F : ndarray
    State and forcing snapshots.
  t, dt : ndarray or float, optional
    Time vector and step size when required.
  n_boxes, n_modes, n_steps, n_ensemble : int, optional
    Model dimensions and ensemble settings.
  w_dict, F_dict : transformer, optional
    Feature maps for EDMD.
  diff_flag, vert_diff_flag, xi, delta, spatial_flag : int or float, optional
    Scenario flags for Budyko-Sellers runs.
  regularize : bool, optional
    Apply ridge regularisation where supported.
  verbose : bool, optional
    Print optimiser diagnostics for modal fits.
  B, exp : any
    Additional placeholders for future extensions.

  Returns
  -------
  operator : ndarray or tuple
    Response function (or pair of matrices) representing the chosen
    emulator.
  """

  # Create emulator based on op_type
  if op_type == 'PS':
    operator = method_Ia_PS(w)

  elif op_type == 'PS_forcing':
    operator = method_Ib_PS_forcing(w, F)

  elif op_type == 'FDT' and xi == 0:
    operator = method_IIa_FDT(n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, spatial_flag=spatial_flag, delta=delta)

  elif op_type == 'FDT':
    operator = method_IIb_FDT(n_ensemble, n_boxes, n_steps, xi, delta, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag)

  elif op_type == 'deconvolve':
    operator = method_III_deconvolve(w, F, dt, regularize)

  elif op_type == 'modal':
    operator = method_IVa_modal(w, F, t, dt, n_modes, n_boxes, num_steps=1000, verbose=verbose)

  elif op_type == 'fit_complex':
    operator = method_IVb_modal_complex(w, F, t, dt, n_modes, n_boxes, num_steps=1000, verbose=verbose)

  elif op_type == 'DMD':
    A_DMD, B_DMD = method_V_DMD(w, F, regularize)
    operator = (A_DMD, B_DMD)

  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = method_VI_EDMD(w, F, w_dict, F_dict, regularize)
    operator = (A_EDMD, B_EDMD)

  else:
    raise ValueError(f'Operator type {op_type} not recognized.')

  return operator

########################
## Emulate a Scenario ##
########################

def emulate_PS(w, pattern):
  """
  Apply pattern scaling to emulate a scenario.

  Parameters
  ----------
  w : ndarray
    Reference field, shape (n_space, n_time).
  pattern : ndarray
    Pattern-scaling coefficients, shape (n_space, 1).

  Returns
  -------
  w_pred : ndarray
    Emulated field, shape (n_space, n_time).
  """

  # Compute global-mean time series and project through pattern
  global_mean = np.mean(w, axis=0).reshape(1,-1)
  w_pred = pattern.T @ global_mean

  return w_pred

def emulate_response(F, G, dt):
  """
  Convolve a response function with a forcing to emulate system output.

  Parameters
  ----------
  F : ndarray
    Forcing time series, shape (n_time,) or (1, n_time).
  G : ndarray
    Response function, shape (n_space, n_time).
  dt : float
    Time step used for scaling the convolution.

  Returns
  -------
  ndarray
    Emulated field, shape (n_space, n_time).
  """

  # Ensure forcing has shape (1, n_time)
  if F.ndim == 1:
    F = F.reshape(1, -1)

  # Build Toeplitz matrix for convolution
  F_toeplitz = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  # Convolve response function with forcing and scale by dt
  return G @ F_toeplitz.T * dt

def emulate_DMD(F, A_DMD, B_DMD, w0, n_steps):
  """
  Emulate a scenario using DMD.

  Parameters
  ----------
  F : ndarray
    Forcing time series, shape (n_space, n_time).
  A_DMD, B_DMD : ndarray
    State-transition and control matrices from DMD.
  w0 : ndarray
    Initial state, shape (n_space,).
  n_steps : int
    Number of time steps to simulate.

  Returns
  -------
  w_pred : ndarray
    Emulated state trajectory, shape (n_space, n_steps).
  """

  # Shape forcing to (n_space, n_time)
  F = check_dim(F)

  # Allocate output array and set initial state
  w_pred = np.zeros((w0.shape[0], n_steps))
  w_pred[:, 0] = w0

  # Time-march using DMD
  for k in range(1, n_steps):
    w_pred[:, k] = A_DMD @ w_pred[:, k-1] + B_DMD @ F[:,k-1]

  return w_pred

def emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F):
  """
  Emulate system using EDMD

  Parameters
  ----------
  F : ndarray
    Forcing time series, shape (n_space, n_time).
  A_EDMD, B_EDMD : ndarray
    State-transition and control matrices from EDMD in feature space.
  w0 : ndarray
    Initial physical state, length n_boxes.
  n_steps : int
    Number of time steps to simulate.
  n_boxes : int
    Number of physical state variables to reconstruct.
  dict_w, dict_F : basis functions
    Feature dictionaries for state and forcing.

  Returns
  -------
  ndarray
    Emulated physical state trajectory, shape (n_boxes, n_steps).
  """

  # Shape forcing to (n_space, n_time) and lift to feauture space
  F = check_dim(F)
  phi0 = dict_w.transform(w0.reshape(1, -1))
  phi0 = phi0.flatten()

  # Allocate arrays for feature space timestepping
  phi_pred = np.zeros((phi0.shape[0], n_steps))
  phi_pred[:, 0] = phi0

  # Allocate arrays for physical space
  w_rec = np.zeros((n_boxes, n_steps))
  w_rec[:, 0] = w0

  # Compute lifted forcings
  F = F.T
  Phi_F = dict_F.transform(F[:-1,:]).T

  # Time-march in feature space using EDMD
  for k in range(1, n_steps):
    phi_pred[:, k] = A_EDMD @ phi_pred[:, k-1] + B_EDMD @ Phi_F[:,k-1]
    w_rec[:, k] = phi_pred[0:n_boxes, k]

  return w_rec

#####################################
## Functions to Evaluate Emulators ##
#####################################

def estimate_w(F, operator, op_type, dt=None, w0=None, n_steps=None, n_boxes=None, dict_w=None, dict_F=None, w=None):
  """
  Route forcing and operator to the appropriate emulator and
  return the estimated state trajectory.

  Parameters
  ----------
  F : ndarray
    Forcing time series.
  operator : ndarray or tuple
    Response function or (A, B) matrices.
  op_type : str
    Operator type {'PS', 'PS_forcing', 'FDT', 'deconvolve',
    'modal', 'fit_complex', 'DMD', 'EDMD'}.
  dt, w0, n_steps, n_boxes, dict_w, dict_F, w : optional
    Auxiliary inputs needed by specific operator types.

  Returns
  -------
  ndarray
    Estimated trajectory of the state variable(s).
  """

  # Select emulator based on operator type
  if op_type == 'PS':
    pattern = operator
    w_est = emulate_PS(w, pattern)

  elif op_type == 'PS_forcing':
    pattern = operator
    w_est = emulate_PS(F, pattern)

  elif op_type == 'deconvolve' or op_type == 'FDT' or op_type == 'modal' or op_type == 'fit_complex':
    w_est = emulate_response(F, operator, dt)

  elif op_type == 'DMD':
    A_DMD, B_DMD = operator
    w_est = emulate_DMD(F, A_DMD, B_DMD, w0, n_steps)

  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = operator
    w_est = emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F)

  else:
    raise ValueError(f'Operator type {op_type} not recognized.')

  return w_est


def calc_RMSE(w_true, w_est):
  """
  Compute the root-mean-square error between true and estimated fields.

  Parameters
  ----------
  w_true : ndarray
    Reference data, shape (n_space, n_time).
  w_est : ndarray
    Estimated data, same shape as `w_true`.

  Returns
  -------
  rmse : ndarray
    RMSE for each spatial location, length n_space.
  """
  return np.sqrt(np.mean((w_true - w_est)**2, axis=1))

def calc_NRMSE(w_true, w_est):
  """
  Normalised root-mean-square error (percentage) between true and
  estimated fields.

  Parameters
  ----------
  w_true : ndarray
    Reference data, shape (n_space, n_time).
  w_est : ndarray
    Estimated data, same shape as `w_true`.

  Returns
  -------
  nrmse : ndarray
    NRMSE for each spatial location, expressed as a percentage of
    the absolute mean true value.
  """
  return calc_RMSE(w_true, w_est)/np.abs(np.mean(w_true, axis=1))*100

def calc_base_NRMSE(error_metrics, scenarios):
  """
  Average NRMSE across space for every train-test scenario pair.

  Parameters
  ----------
  error_metrics : dict
    Nested dictionary where error_metrics[train][test] holds
    NRMSE arrays for each train/test combination.
  scenarios : iterable
    List or tuple of scenario names used as keys.

  Returns
  -------
  dict
    Dictionary NRMSE_base[train][test] with mean NRMSE values
    (train == test pairs are omitted).
  """

  # Initialize output container
  NRMSE_base = {}

  # Loop over training scenarios
  for train in scenarios:
    NRMSE_base[train] = {}

    # Loop over testing scenarios
    for test in scenarios:
      # Skip trivial case
      if train == test:
        continue

      # Save mean NRMSE
      NRMSE_base[train][test] = np.mean(error_metrics[train][test])

  return NRMSE_base

############################################
## Create Emulators and Emulate Scenarios ##
############################################

def emulate_scenarios(op_type, scenarios=None, outputs=None, forcings=None, w0=None, t=None, dt=None, n_steps=None, n_boxes=None,
                        w_dict=None, F_dict=None, n_modes=None, verbose=True, diff_flag=0, vert_diff_flag=0, B=None, xi=0, n_ensemble=None,
                        delta=0, t_range=None, regularize=False, spatial_flag=0, exp=None):
  """
  Build an emulator for each training scenario and evaluate its performance
  across all test scenarios.

  Parameters
  ----------
  op_type : str
    Operator choice {'PS', 'PS_forcing', 'FDT', 'deconvolve', 'modal',
    'fit_complex', 'DMD', 'EDMD'}.
  scenarios : list
    Scenario names used as keys into `forcings` and `outputs`.
  outputs, forcings : dict
    Dicts mapping scenario -> ndarray of model outputs / forcings.
  w0, t, dt, n_steps, n_boxes, w_dict, F_dict, n_modes :
    Auxiliary arguments forwarded to the individual emulator builders.
  diff_flag, vert_diff_flag, B, xi, n_ensemble, delta, spatial_flag, exp :
    Budyko-Sellers and other configuration flags.
  t_range : slice or ndarray, optional
    Sub-range of time indices to use for training data.
  regularize : bool, optional
    Apply ridge regularisation where supported.
  verbose : bool, optional
    Print progress and NRMSE values.

  Returns
  -------
  operator : dict
    Trained operator(s) keyed by training scenario (or single entry for FDT).
  w_pred : dict
    Nested dict of predicted trajectories [train][test] (or flat dict for FDT).
  error_metrics : dict
    Corresponding NRMSE arrays.
  """

  # Initialize containers
  operator, w_pred, error_metrics = {}, {}, {}

  # Separate treament for FDT
  if op_type == 'FDT':
    operator = create_emulator(op_type, None, None, n_boxes=n_boxes, diff_flag=diff_flag,
                               vert_diff_flag=vert_diff_flag, spatial_flag=spatial_flag, delta=delta)
    if verbose:
      print(f'Train: Impulse Forcing - NRMSE')

    # Evaluate FDT operator across all scenarios
    for scen2 in scenarios:
      F2 = forcings[scen2]
      w_true_2 = outputs[scen2]
      w_pred[scen2] = estimate_w(F2, operator, op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
      error_metrics[scen2] = calc_NRMSE(w_true_2, w_pred[scen2])
      if verbose:
        print(f'\tTest: {scen2} - {error_metrics[scen2]}')

  # All other operators
  else:
    for scen1 in scenarios:
      # Containers for each training scenario
      w_pred[scen1], error_metrics[scen1] = {}, {}
      if verbose:
        print(f'Train: {scen1} - NRMSE')

      F1, w_true_1 = forcings[scen1], outputs[scen1]
      F1, w_true_1 = check_dim(F1), check_dim(w_true_1)

      # Optional: crop a specific range of time for training
      if t_range is not None:
        F1, w_true_1 = F1[:,t_range], w_true_1[:,t_range]

      # Build emulator for current training scenario
      operator[scen1] = create_emulator(op_type, w_true_1, F1, t=t, dt=dt, n_boxes=n_boxes, w_dict=w_dict,
                                       F_dict=F_dict, n_modes=n_modes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag,
                                       B=B, xi=xi, n_ensemble=n_ensemble, delta=delta, n_steps=n_steps, regularize=regularize,
                                       verbose=verbose, exp=exp)

      # Test emulator against every scenario
      for scen2 in scenarios:
        F2, w_true_2 = forcings[scen2], outputs[scen2]
        F2, w_true_2 = check_dim(F2), check_dim(w_true_2)
        w_pred[scen1][scen2] = estimate_w(F2, operator[scen1], op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict, w=w_true_2)
        error_metrics[scen1][scen2] = calc_NRMSE(w_true_2, w_pred[scen1][scen2])

        if verbose:
          print(f'\tTest: {scen2} - {error_metrics[scen1][scen2]}')

  return operator, w_pred, error_metrics

###############################
## Ensemble Helper Functions ##
###############################

def evaluate_ensemble(op_type, w_ensemble, w_mean, F_ensemble, F_mean, scenarios,
                      n_ensemble, n_choices, dt=None, t=None, n_modes=None, w0=None,
                      n_steps=None, n_boxes=None, w_dict=None, F_dict=None, G_FDT=None,
                      cubLor=False, rho_base=None, t_vec=None, baseline_mean=None,
                      z_base_ensemble=None, z_pert_ensemble=None, delta=None):
  """
  Assess emulator skill as the number of training realisations varies.

  Parameters
  ----------
  op_type : str
    Operator type {'PS', 'deconvolve', 'modal', 'DMD', 'EDMD', 'FDT'}.
  w_ensemble, F_ensemble : dict
    Scenario -> ndarray of individual realisations
  w_mean, F_mean : dict
    Scenario -> ensemble-mean field / forcing.
  scenarios : list
    Scenario names.
  n_ensemble : int
    Total ensemble members available.
  n_choices : int
    Random draws for each subset size.
  dt, t, n_modes, w0, n_steps, n_boxes, w_dict, F_dict, G_FDT :
    Auxiliary data needed by specific operator types.
  cubLor, rho_base, t_vec, baseline_mean, z_base_ensemble,
  z_pert_ensemble, delta : optional
    Flags and data for the cubic Lorenz test case.

  Returns
  -------
  dict
    NRMSE values aggregated over subset size and scenario pairs.
  """

  # Random seed for cubic Lorenz
  if cubLor:
    rng = np.random.default_rng()

  # Output container
  NRMSE_all = {}

  # Separate treatment for cubic Lorenz FDT
  if op_type == 'FDT' and cubLor:
    for test in scenarios:
      NRMSE_all[test] = []

    for n in range(1, n_ensemble + 1, n_ensemble//50):
      NRMSE_temp = {}
      for test in scenarios:
        NRMSE_temp[test] = []

      for _ in range(10):
        # Choose n ensemble members
        idx = rng.choice(n_ensemble, size=n, replace=False)
        z_choice_base = z_base_ensemble[idx]
        z_choice_pert = z_pert_ensemble[idx]
        G_temp = np.mean(z_choice_pert - z_choice_base, axis=0) / delta

        for test in scenarios:
          # Calculate error tested against the full ensemble (i.e. avg. forcing)
          pred_anom_temp = np.convolve(G_temp.flatten(), (F_mean[test] - rho_base), mode='full')[:len(t_vec)] * dt
          w_pred_temp = pred_anom_temp + baseline_mean[2]
          NRMSE_temp[test].append(np.mean(calc_NRMSE(w_mean[test].reshape(1,-1), w_pred_temp.reshape(1,-1))))

      # Save average error over trials
      for test in scenarios:
        NRMSE_all[test].append(np.mean(NRMSE_temp[test]))

  # All other cases
  else:
    for train in scenarios:
      NRMSE_all[train] = {}
      if cubLor:
        w_train = w_ensemble[train]
      else:
        w_and_F = list(zip(w_ensemble[train], F_ensemble[train]))

      if train == 'Mid. Emissions' or train == 'Overshoot':
        lam = 1e1
      else:
        lam = 1e-4

      # FDT for non-Lorenz case
      if op_type == 'FDT' and not cubLor:
        NRMSE_all[train] = []
        w_pred_temp = emulate_response(F_mean[train], G_FDT, dt)

        for _ in range(n_ensemble):
          NRMSE_all[train].append(np.mean(calc_NRMSE(w_mean[train], w_pred_temp)))

        continue

      # Initialize containers
      for test in scenarios:
        if train == test:
          continue
        NRMSE_all[train][test] = []

      for n in range(1, n_ensemble + 1, n_ensemble//50):
        NRMSE_temp = {}
        for test in scenarios:
          if train == test:
            continue

          NRMSE_temp[test] = []

        for _ in range(n_choices):
          # Choose n ensemble members
          if cubLor:
            idx = rng.choice(n_ensemble, size=n, replace=False)
            w_choice = w_train[idx]
          else:
            temp_choice = random.sample(w_and_F, n)
            w_choice, F_choice = zip(*temp_choice)

          # Calculate operator
          if op_type == 'PS' and cubLor:
            operator_temp = method_Ib_PS_forcing(np.mean(w_choice, axis=0), F_mean[train])
          elif op_type == 'PS':
            operator_temp = method_Ia_PS(np.mean(w_choice, axis=0))
          elif op_type == 'deconvolve' and cubLor:
            G_temp = method_III_deconvolve(np.mean(w_choice, axis=0)[0:501], F_mean[train][0:501], dt)
          elif op_type == 'deconvolve':
            G_temp = method_III_deconvolve(np.mean(w_choice, axis=0), np.mean(F_choice, axis=0), dt, regularize=True)
          elif op_type == 'modal' and cubLor:
            G_temp = method_IVb_modal_complex(np.mean(w_choice, axis=0)[0:501], F_mean[train][0:501], t_vec[0:501], dt, n_modes, n_boxes)
          elif op_type == 'modal':
            G_temp = method_IVa_modal(np.mean(w_choice, axis=0), np.mean(F_choice, axis=0), t, dt, n_modes, n_boxes)
          elif op_type == 'DMD' and cubLor:
            A_temp, B_temp = method_V_DMD(np.mean(w_choice, axis=0), F_mean[train])
          elif op_type == 'DMD':
            A_temp, B_temp = method_V_DMD(np.mean(w_choice, axis=0), np.mean(F_choice, axis=0), regularize=True, lam=lam)
          elif op_type == 'EDMD' and cubLor:
            A_temp, B_temp = method_VI_EDMD(np.mean(w_choice, axis=0), F_mean[train], w_dict, F_dict)
          elif op_type == 'EDMD':
            A_temp, B_temp = method_VI_EDMD(np.mean(w_choice, axis=0), np.mean(F_choice, axis=0), w_dict, F_dict, regularize=True, lam=lam)
          else:
            raise ValueError(f'Operator type {op_type} not recognized.')

          for test in scenarios:
            if train == test:
              continue

            # Calculate error tested against the full ensemble (i.e. avg. forcing)
            if op_type == 'PS' and cubLor:
              w_pred_temp = operator_temp @ F_mean[test].reshape(1,-1)
            elif op_type == 'PS':
              w_pred_temp = emulate_PS(w_mean[test], operator_temp)
            elif (op_type == 'deconvolve' or op_type == 'modal') and cubLor:
              pred_anom_temp = np.convolve(G_temp.flatten(), (F_mean[test] - rho_base), mode='full')[:len(t_vec)] * dt
              w_pred_temp = pred_anom_temp + baseline_mean[2]
            elif op_type == 'deconvolve' or op_type == 'modal':
              w_pred_temp = emulate_response(F_mean[test], G_temp, dt)
            elif op_type == 'DMD':
              w_pred_temp = emulate_DMD(F_mean[test], A_temp, B_temp, w0, n_steps)
            elif op_type == 'EDMD':
              w_pred_temp = emulate_EDMD(F_mean[test], A_temp, B_temp, w0, n_steps, n_boxes, w_dict, F_dict)

            if cubLor:
              NRMSE_temp[test].append(np.mean(calc_NRMSE(w_mean[test].reshape(1,-1), w_pred_temp)))
            elif cubLor and (op_type == 'FDT' or op_type == 'deconvolve'):
              NRMSE_temp[test].append(np.mean(calc_NRMSE(w_mean[test].reshape(1,-1), w_pred_temp.reshape(1,-1))))
            else:
              NRMSE_temp[test].append(np.mean(calc_NRMSE(w_mean[test], w_pred_temp)))

        # Save average error over trials
        for test in scenarios:
          if train == test:
            continue
          NRMSE_all[train][test].append(np.mean(NRMSE_temp[test]))

  return NRMSE_all

##############################
## General Helper Functions ##
##############################

def check_dim(var, transp=False):
  """
  Ensure an input array is 2-D.

  Parameters
  ----------
  var : ndarray
    Input array, 1-D or 2-D.
  transp : bool, optional
    If True, reshape a 1-D vector to a column (n, 1);
    otherwise reshape to a row (1, n).

  Returns
  -------
  ndarray
    Original array if already 2-D, else reshaped 2-D view.
  """

  # Check dimensionality
  if var.ndim == 1:
    if transp:
      return var.reshape(-1, 1)
    else:
      return var.reshape(1, -1)
  return var

def save_results(metrics, name):
  """
  Save a metrics dictionary to disk as a pickle file.

  Parameters
  ----------
  metrics : dict
    Data to be written.
  name : str
    Filename (without extension) stored under 'Results/'.

  Returns
  -------
  None
  """

  with open(f'Results/{name}.pkl', 'wb') as file:
    pickle.dump(metrics, file)
  return

def open_results(name):
  """
  Load a pickled metrics object from disk.

  Parameters
  ----------
  name : str
    Filename (without “.pkl”) stored in the 'Results/' directory.

  Returns
  -------
  dict
    Metrics dictionary retrieved from disk.
  """

  with open(f'Results/{name}.pkl', 'rb') as file:
    metric = pickle.load(file)
  return metric

def get_regularization(w, F):
  """
  Estimate ridge regularization hyper-parameters by
  minimising a data-misfit cost.

  Parameters
  ----------
  w : ndarray
    Target response, shape (n_space, n_time).
  F : ndarray
    Forcing time series, shape (1, n_time) or (n_time,).

  Returns
  -------
  ndarray
    Optimized hyper-parameters [sig2, lam2].
  """

  # Random initial guess for the two hyper-parameters
  init_params = np.random.rand(2)

  # Toeplitz representation of forcing for convolution
  F_toep = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))

  # Optimization of the hyper-parameter cost function
  res = minimize(fit_opt_hyper,
                 init_params,
                 args=(w, F_toep),
                 method='L-BFGS-B')

  return res.x

def fit_opt_hyper(params, w, F_toep):
  """
  Negative log-evidence objective for tuning regularisation
  hyper-parameters (sig^2, lam^2²).

  Parameters
  ----------
  params : array-like
    Two-element vector [sig2, lam2] to be optimised.
  w : ndarray
    Response data, shape (n_space, n_time).
  F_toep : sparse matrix
    Toeplitz forcing operator (n_time x n_time).

  Returns
  -------
  float
    Value of the objective to minimise.
  """

  # Unpack hyper-parameters
  sig2, lam2 = params

  # Build covariance matrix
  n_x, n_t = w.shape
  Sig = sig2*np.eye(n_t) + lam2*(F_toep @ F_toep.T)

  # Log-determinant of sigma; invalid if sign <= 0
  sign, logdet_Sig = np.linalg.slogdet(Sig)
  if sign <= 0:
    return np.inf

  # Quadratic form
  Sig_inv_w = np.linalg.solve(Sig, w.T)
  quadratic_term = np.sum(w * Sig_inv_w.T)

  # Log-evidence (marginal likelihood) and its negative
  log_evidence_value = -0.5 * (n_x * logdet_Sig + quadratic_term)
  return -log_evidence_value

def plot_true_pred(T_true, T_pred, experiments, operator=None):
  """
  Visualize emulator skill by plotting true vs. predicted temperature
  trajectories across all train-test experiment pairs.

  Parameters
  ----------
  T_true : dict
    Scenario -> ndarray (n_space, n_time) of reference temperatures.
  T_pred : dict
    Nested dict T_pred[train][test] with emulator predictions.
  experiments : list
    Ordered list of scenario names to plot.
  operator : str, optional
    Name of operator (e.g. 'PS') for title & special handling.

  Returns
  -------
  None
  """

  # Initialize subplots
  n_exp = len(experiments)
  fig, ax = plt.subplots(n_exp, n_exp, figsize=(12,12), constrained_layout=True, sharex='col', sharey='row')

  # Iterate over all train-test pairs and plot
  for i, exp1 in enumerate(experiments):
    for j, exp2 in enumerate(experiments):
      T_true_temp, T_pred_temp = T_true[exp2].T, T_pred[exp1][exp2].T
      T_true_temp, T_pred_temp = check_dim(T_true_temp, transp=True), check_dim(T_pred_temp, transp=True)

      n_boxes = T_true_temp.shape[1]

      if operator == 'PS':
        ax[i,j].plot(np.mean(T_true_temp,axis=1), lw=2, c='k', alpha=0.75)

      for k in range(n_boxes):
        ax[i,j].plot(T_true_temp[:,k], lw=2, c=cm.batlowS(k + 3), alpha=0.5)
        ax[i,j].plot(T_pred_temp[:,k], lw=2, c=cm.batlowS(k + 3), ls='-.')

  # Add experiment names as column titles (train)
  for j, train_name in enumerate(experiments):
    ax[n_exp-1, j].set_xlabel(train_name, fontsize=14, labelpad=10)

  # Add experiment names as row labels (test)
  for i, test_name in enumerate(experiments):
    ax[i, 0].set_ylabel(test_name, fontsize=14, rotation=90, labelpad=10, va="center")

  # Add title
  fig.suptitle(f'Emulator Performance - {abbrev_to_full[operator]}')

  # Label overall X and Y axes
  fig.text(0.5, -0.02, "Train Scenario", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(-0.02, 0.5, "Test Scenario", ha="center", va="center", fontsize=18, fontweight="bold", rotation=90)

  return

def plot_true_pred_FDT(T_true, T_pred, experiments):
  """
  Plot true vs. predicted temperature trajectories for an FDT-based
  emulator, with one panel per test scenario.

  Parameters
  ----------
  T_true : dict
    Scenario -> ndarray (n_space, n_time) of reference temperatures.
  T_pred : dict
    Scenario -> ndarray of emulator predictions (same shapes).
  experiments : list
    Ordered list of scenario names to display.

  Returns
  -------
  None
  """

  # Initialize subplots
  n_exp = len(experiments)
  fig, ax = plt.subplots(1, n_exp, figsize=(12,4), constrained_layout=True)

  # Iterate over all experiments
  for i, exp in enumerate(experiments):
    T_true_temp, T_pred_temp = T_true[exp].T, T_pred[exp].T
    n_boxes = T_true_temp.shape[1]
    for k in range(n_boxes):
      ax[i].plot(T_true_temp[:,k], lw=2, c=cm.batlowS(k + 3), alpha=0.5)
      ax[i].plot(T_pred_temp[:,k], lw=2, c=cm.batlowS(k + 3), ls='-.')

  # Add experiment names as row labels (test)
  for i, test_name in enumerate(experiments):
    ax[i].set_xlabel(test_name, fontsize=14, labelpad=10, va="center")

  # Add title
  fig.suptitle(f'Emulator Performance - {abbrev_to_full['FDT']}')

  # Label overall X and Y axes
  fig.text(0.5, -0.02, "Test Scenario", ha="center", va="center", fontsize=18, fontweight="bold")

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
      Phi.append(exp_term)
      Phi.append(exp_term * np.cos(omega * xi))
      Phi.append(exp_term * np.sin(omega * xi))

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
    X_dot[:-1, :] = np.diff(X, axis=0)

    # For the last point, use a backward difference
    X_dot[-1, :] = X[-1, :] - X[-2, :]

    # Concatenate the original data and its derivative side-by-side
    return np.hstack([X, X_dot])

############
## Useful ##
############

abbrev_to_full = {
  'PS':'Method I: Pattern Scaling',
  'PS_forcing':'Method I: Pattern Scaling w/ Forcing',
  'FDT':'Method II: Fluctuation Dissipation Theorem',
  'deconvolve':'Method III: Deconvolution',
  'modal':'Method IV: Modal Fitting',
  'DMD':'Method V: Dynamic Mode Decomposition',
  'EDMD':'Method VI: Extended DMD'
}