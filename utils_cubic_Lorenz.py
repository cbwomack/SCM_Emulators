# Required imports
import numpy as np
import utils_emulator

def Lorenz_rho(t, omega=1, exp=0):
  """
  Time-varying rho parameter for cubic Lorenz systems.

  Parameters
  ----------
  t : float or ndarray
    Time variable.
  omega : float, optional
    Frequency used by some scenarios (default 1).
  exp : int, optional
    Scenario flag
    0 = abrupt, 1 = high, 2 = mid, 3 = overshoot,
    4 = sinusoid, 5 = noise, 6 = constant.

  Returns
  -------
  float or ndarray
    rho for the selected scenario.
  """
  # Calculate rho for cubic Lorenz system
  if exp == 0: # Abrupt
    return 45 + 17*np.tanh(omega*(t - 10))
  elif exp == 1: # High Emissions
    return 28 + 30/(np.exp(250/50))*np.exp(t/50)
  elif exp == 2: # Plateau
    return 40 + 12*np.tanh(1/50*(t - 150))/np.tanh(5)
  elif exp == 3: # Overshoot
    return 28 + 30*np.exp(-np.power(t - 200,2)/(2*50**2))
  elif exp == 4: # Sinusoid
    return 60 + 30*np.sin(omega*t)
  elif exp == 5: # Noise
    return (0.25*np.exp(-t/5) + 0.75*np.exp(-t/0.05))*np.cos(2*np.pi*t/0.57)
  elif exp == 6:
    return 28 # Constant initial condition
  else:
    raise ValueError('Error, unrecognized experiment.')

def cub_lorenz(state, rho, sigma, alpha, beta):
  """
  Compute the instantaneous derivatives for the cubic Lorenz system.

  Parameters
  ----------
  state : ndarray
    Current states, shape (n, 3) with columns (x, y, z).
  rho, sigma, alpha, beta : float
    System parameters.

  Returns
  -------
  ndarray
    Derivatives (dx, dy, dz) for each state, same shape as `state`.
  """

  # Unpack coordinates
  x, y, z = state[:, 0], state[:, 1], state[:, 2]

  # Evaluate cubic Lorenz RHS
  dx = sigma * (y - x)
  dy = -(z + alpha*np.power(z,3))*x + rho*x - y
  dz = x * y - beta * z
  return np.stack([dx, dy, dz], axis=1)

def rk4_step(state, rho, sigma, alpha, beta, dt):
  """
  Advance the cubic Lorenz system one time step using
  fourth-order Runge-Kutta integration.

  Parameters
  ----------
  state : ndarray
    Current states, shape (n, 3).
  rho, sigma, alpha, beta : float
    System parameters.
  dt : float
    Time step.

  Returns
  -------
  ndarray
    Updated states after one RK4 step, shape (n, 3).
  """

  # Take derivates in stages
  k1 = cub_lorenz(state,               rho, sigma, alpha, beta)
  k2 = cub_lorenz(state + 0.5*dt*k1,   rho, sigma, alpha, beta)
  k3 = cub_lorenz(state + 0.5*dt*k2,   rho, sigma, alpha, beta)
  k4 = cub_lorenz(state +       dt*k3, rho, sigma, alpha, beta)

  # Combine steps for RK4
  return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def gen_rho(t_vec, scenarios):
  """
  Generate time-series of rho for multiple scenarios.

  Parameters
  ----------
  t_vec : ndarray
    Array of time points.
  scenarios : list
    Ordered list of scenario names (used only for keys).

  Returns
  -------
  dict
    Dictionary mapping each scenario -> rho array.
  """

  # Output container
  rho = {}

  # Loop over scenarios and generate rho array
  for i, scen in enumerate(scenarios):
    rho[scen] = np.zeros(len(t_vec))
    for j, t in enumerate(t_vec):
      rho[scen][j] = Lorenz_rho(t, exp=i)

  return rho

def spin_up(n_ensemble, rho_base, sigma, alpha, beta, dt, eps, save=False):
  """
  Spin up an ensemble of cubic Lorenz initial states.

  Parameters
  ----------
  n_ensemble : int
    Number of ensemble members.
  rho_base, sigma, alpha, beta : float
    Lorenz system parameters.
  dt : float
    Time step for integration.
  eps : float
    Noise amplitude for stochastic forcing.
  save : bool, optional
    If True, pickle the spun-up states and their mean.

  Returns
  -------
  state0 : ndarray
    Spun-up states, shape (n_ensemble, 3).
  baseline_mean : ndarray
    Ensemble-mean state after spin-up, length 3.
  """

  # Initialize random states
  state0 = np.random.normal(0.0, 5.0, size=(n_ensemble, 3))

  # Spin up integration with stochastic forcing
  warm_steps = 5_000
  for _ in range(warm_steps):
    dW      = np.random.normal(0.0, 2, size=state0.shape)
    state0  = rk4_step(state0, rho_base, sigma, alpha, beta, dt) + eps * np.sqrt(dt) * dW

  # Compute ensemble mean
  baseline_mean = state0.mean(axis=0)

  # Optional: save outputs
  if save:
    utils_emulator.save_results(state0, 'exp4_state0')
    utils_emulator.save_results(baseline_mean, 'exp4_baseline_mean')

  return state0, baseline_mean

def run_Lorenz(scenarios, state0, n_ensemble, n_steps_scen, rho, sigma, alpha, beta, dt, eps, save=False, verbose=False):
  """
  Integrate a stochastic cubic Lorenz ensemble for multiple forcing
  scenarios.

  Parameters
  ----------
  scenarios : list
    Scenario names that map to entries in `rho`.
  state0 : ndarray
    Initial ensemble states, shape (n_ensemble, 3).
  n_ensemble : int
    Number of ensemble members.
  n_steps_scen : int
    Time steps per scenario.
  rho : dict
    Scenario -> rho array.
  sigma, alpha, beta : float
    Lorenz system parameters.
  dt : float
    Time step.
  eps : float
    Noise amplitude for stochastic forcing.
  save : bool, optional
    If True, pickle the full trajectories and their means.
  verbose : bool, optional
    If True, print progress information.

  Returns
  -------
  state_ensemble : dict
    Scenario -> ndarray (n_ensemble, n_steps_scen, 3) of trajectories.
  state_mean : dict
      Scenario -> ndarray (n_steps_scen, 3) of ensemble means.
  """

  # Output containers
  state_ensemble, state_mean = {}, {}

  # Loop over scenarios
  for scen in scenarios:
    if verbose:
      print(scen)

    # Baseline and perturbed ensembles share identical initial states
    state = state0.copy()

    # Storage for means
    state_ensemble[scen] = np.zeros((n_ensemble, n_steps_scen, 3))
    state_ensemble[scen][:,0,:] = state
    state_mean[scen] = np.zeros((n_steps_scen, 3))
    state_mean[scen][0] = state.mean(axis=0)

    # Time integration
    for n in range(1, n_steps_scen):
      rho_t = rho[scen][n]
      dW = np.random.normal(0.0, 1, size=state0.shape)

      # RK4 deterministic advance + stochastic term
      state_ensemble[scen][:,n,:] = rk4_step(state_ensemble[scen][:,n-1,:], rho_t, sigma, alpha, beta, dt) + eps * np.sqrt(dt) * dW
      state_mean[scen][n] = state_ensemble[scen][:,n,:].mean(axis=0)

      if n % 10_000 == 0 and verbose:
        print(f'\t{n}')

    # Optional: save outputs
    if save:
      utils_emulator.save_results(state_ensemble, 'exp4_ensemble')
      utils_emulator.save_results(state_mean, 'exp4_mean')

  return state_ensemble, state_mean

def get_z_vals(scenarios, open=False, state_ensemble=None, state_mean=None):
  """
  Extract the z-coordinate trajectories (3rd state variable) from
  stored or provided Lorenz ensemble data.

  Parameters
  ----------
  scenarios : list
    Scenario names present in the data.
  open : bool, optional
    If True, load data from disk; otherwise use supplied arrays.
  state_ensemble, state_mean : dict, optional
    Pre-loaded dictionaries from `run_Lorenz`. Ignored if `open` is True.

  Returns
  -------
  z_ensemble : dict
    Scenario -> ndarray (n_ensemble, n_steps) of z trajectories.
  z_mean : dict
    Scenario -> ndarray (n_steps,) of ensemble-mean z.
  z_std : dict
    Scenario -> ndarray (n_steps,) of ensemble-std dev of z.
  """

  # Load data from disk when requested
  if open:
    state_ensemble = utils_emulator.open_results('exp4_ensemble')
    state_mean = utils_emulator.open_results('exp4_mean')

  # Containers for z statistics
  z_ensemble, z_mean, z_std = {}, {}, {}

  # Loop over scenarios and slice the z component (index 2)
  for scen in scenarios:
    z_ensemble[scen] = state_ensemble[scen][:,:,2].copy()
    z_mean[scen] = state_mean[scen][:,2].copy()
    z_std[scen] = np.std(state_ensemble[scen][:,:,2],axis=0)

  return z_ensemble, z_mean, z_std

def run_Lorenz_pert(state0, n_ensemble, n_steps_pert, rho_base, sigma, alpha, beta, dt, eps):
  """
  Integrate paired baseline and perturbed cubic-Lorenz ensembles to
  estimate the response function using the FDT.

  Parameters
  ----------
  state0 : ndarray
    Initial states, shape (n_ensemble, 3).
  n_ensemble : int
    Number of ensemble members.
  n_steps_pert : int
    Integration steps for the perturbation experiment.
  rho_base, sigma, alpha, beta : float
    Lorenz system parameters.
  dt : float
    Time step.
  eps : float
    Stochastic-noise amplitude.

  Returns
  -------
  R_FDT : ndarray
    Response estimate, shape (n_steps_pert, 3).
  state_base_ensemble : ndarray
    Baseline trajectories, shape (n_ensemble, n_steps_pert, 3).
  state_pert_ensemble : ndarray
    Perturbed trajectories, same shape as above.
  """

  # Allocate arrays for baseline and perturbed ensembles
  state_base_ensemble, state_pert_ensemble = np.zeros((n_ensemble, n_steps_pert, 3)), np.zeros((n_ensemble, n_steps_pert, 3))

  # Baseline and perturbed ensembles share identical initial states
  delta = 50 * dt
  state_base_ensemble[:,0,:]  = state0.copy()
  state_pert_ensemble[:,0,:]  = state0.copy()
  state_pert_ensemble[:,0,1] += delta * state_base_ensemble[:,0,0]

  # Storage for means
  mean_base = np.zeros((n_steps_pert, 3))
  mean_pert = np.zeros_like(mean_base)
  mean_base[0] = state_base_ensemble[:,0,:].mean(axis=0)
  mean_pert[0] = state_pert_ensemble[:,0,:].mean(axis=0)

  # Time integration
  for n in range(1, n_steps_pert):
    # identical noise for baseline & perturbed trajectories
    dW = np.random.normal(0.0, 1, size=state0.shape)

    # RK4 deterministic advance + identical stochastic kick
    state_base_ensemble[:,n,:] = rk4_step(state_base_ensemble[:,n-1,:], rho_base, sigma, alpha, beta, dt) + eps * np.sqrt(dt) * dW
    state_pert_ensemble[:,n,:] = rk4_step(state_pert_ensemble[:,n-1,:], rho_base, sigma, alpha, beta, dt) + eps * np.sqrt(dt) * dW

    mean_base[n] = state_base_ensemble[:,n,:].mean(axis=0)
    mean_pert[n] = state_pert_ensemble[:,n,:].mean(axis=0)

  # Approximate response function
  R_FDT = (mean_pert - mean_base) / delta

  return R_FDT, state_base_ensemble, state_pert_ensemble