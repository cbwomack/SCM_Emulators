from venv import create
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize
import random
import BudykoSellers

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
  if exp == '2xCO2':
    n_t = len(t)
    F = abrupt_2xCO2(n_boxes, n_t)

  elif exp == 'High Emissions':
    F = high_emissions(8.5, t, t_end, t_star, n_boxes)

  elif exp == 'Overshoot':
    F = overshoot(t, n_boxes)

  return F

#####################################
## Functions to Diagnose Emulators ##
#####################################

def method_1a_DMD(w, F):
  # Calculate L using DMD
  # Assume F is of size (nx, nt)
  Omega = np.concatenate([w[:,:-1],F[:,:-1]])
  L = w[:,1:] @ np.linalg.pinv(Omega)
  n = len(w)

  A_DMD, B_DMD = L[:,:n], L[:,n:]

  return A_DMD, B_DMD


def method_1b_EDMD(w, F, dict_w, dict_F):
  # Calculate K using EDMD
  F = F.T
  #F = np.hstack((F.T, np.zeros((F.shape[1], 2))))

  Phi_F = dict_F.transform(F[:-1,:])
  Phi_w = dict_w.transform(w[:,:-1].T)
  Phi_wprime = dict_w.transform(w[:,1:].T)

  Omega = np.concatenate([Phi_w.T,Phi_F.T])
  K = Phi_wprime.T @ np.linalg.pinv(Omega)
  n = len(Phi_wprime.T)

  A_EDMD, B_EDMD = K[:,:n], K[:,n:]

  return A_EDMD, B_EDMD

def method_2a_direct(n_boxes, diff_flag=0, vert_diff_flag=0, xi=0):
  # Calculate G directly
  full_output = BudykoSellers.Run_Budyko_Sellers(exp_flag=3, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi)
  G_direct = np.squeeze(full_output['T_ts'])[0:n_boxes,:]

  return G_direct

def method_2b_FDT(n_ensemble, n_boxes, n_steps, xi, delta, exp_flag=0, diff_flag=0, vert_diff_flag=0):
  # Calculate G from an ensemble using the FDT
  w, w_delta = np.zeros((n_ensemble, n_boxes, n_steps)), np.zeros((n_ensemble, n_boxes, n_steps))

  # Run n_ensemble number of ensemble members
  for n in range(n_ensemble):
    # Run unperturbed scenario
    full_output_unperturbed = BudykoSellers.Run_Budyko_Sellers(exp_flag=exp_flag, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi)
    w[n,:,:] = np.squeeze(full_output_unperturbed['T_ts'])[0:n_boxes,:]
    noise_ts = full_output_unperturbed['noise_ts']

    # Run perturbed scenario
    full_output_perturbed = BudykoSellers.Run_Budyko_Sellers(exp_flag=exp_flag, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi, delta=delta, noise_ts=noise_ts)
    w_delta[n,:,:] = np.squeeze(full_output_perturbed['T_ts'])[0:n_boxes,:]

  # Take ensemble average divided by magnitude of perturbation
  G_FDT = np.mean(w_delta - w, axis=0)/delta

  return G_FDT

def method_3a_deconvolve(w, F, dt, regularize=False):
  # Calculate G using deconvolution
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
                    vert_diff_flag=0, B=None, xi=0, n_ensemble=None, n_steps=None, delta=0):
  if op_type == 'DMD':
    A_DMD, B_DMD = method_1a_DMD(w, F)
    operator = (A_DMD, B_DMD)

  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = method_1b_EDMD(w, F, w_dict, F_dict)
    operator = (A_EDMD, B_EDMD)

  elif op_type == 'deconvolve':
    operator = method_3a_deconvolve(w, F, dt)

  elif op_type == 'direct':
    operator = method_2a_direct(n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag, xi=xi)

  elif op_type == 'FDT':
    operator = method_2b_FDT(n_ensemble, n_boxes, n_steps,  xi, delta, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag)

  elif op_type == 'fit':
    operator = method_4a_fit(w, F, t, dt, n_modes, n_boxes, B)

  elif op_type == 'fit_DMD':
    A_DMD, B_DMD = method_1a_DMD(w, F)
    operator = method_4a_fit(w, F, t, dt, n_modes, n_boxes, B, A_DMD, B_DMD)

  else:
    raise ValueError('Operator type {op_type} not recognized.')

  return operator

########################
## Emulate a Scenario ##
########################

def emulate_DMD(F, A_DMD, B_DMD, w0, n_steps):
  # Emulate a scenario with DMD
  w_pred = np.zeros((w0.shape[0], n_steps))
  w_pred[:, 0] = w0

  for k in range(1, n_steps):
    w_pred[:, k] = A_DMD @ w_pred[:, k-1] + B_DMD @ F[:,k-1]

  return w_pred

def emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F):

  # x0 is shape (3,)
  # Convert to (1,3) for dictionary.transform
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

    # Reconstruct the state from the first 3 dimensions
    w_rec[:, k] = phi_pred[1:n_boxes+1, k]  # Depends on how dictionary is defined

  return w_rec

def emulate_response(F, G, dt):
  # Emulate a scenario with a response function
  F_toeplitz = sparse.csr_matrix(toeplitz(F[0,:], np.zeros_like(F[0,:])))
  return G @ F_toeplitz.T * dt

#####################################
## Functions to Evaluate Emulators ##
#####################################

def estimate_w(F, operator, op_type, dt=None, w0=None, n_steps=None, n_boxes=None, dict_w=None, dict_F=None):
  # Estimate variable of interest given an initial
  # condition and forcing

  if op_type == 'DMD':
    A_DMD, B_DMD = operator
    w_est = emulate_DMD(F, A_DMD, B_DMD, w0, n_steps)
  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = operator
    w_est = emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, n_boxes, dict_w, dict_F)
  elif op_type == 'deconvolve' or op_type == 'direct' or op_type == 'FDT' or op_type == 'fit' or op_type == 'fit_DMD':
    w_est = emulate_response(F, operator, dt)
  else:
    raise ValueError('Operator type {op_type} not recognized.')

  return w_est

def calc_L2(w_true, w_est):
  # Estimate L2 error between emulator and ground truth
  return np.linalg.norm(w_true - w_est)

def emulate_experiments(op_type, experiments=None, outputs=None, forcings=None, w0=None, t=None, dt=None, n_steps=None, n_boxes=None,
                        w_dict=None, F_dict=None, n_modes=None, verbose=True, diff_flag=0, vert_diff_flag=0, B=None, xi=0, n_ensemble=None, delta=0):
  operator, w_pred, L2 = {}, {}, {}

  if op_type == 'direct':
    operator = create_emulator(op_type, None, None, n_boxes=n_boxes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag)
    if verbose:
      print(f'Train: Impulse Forcing - L2 Error')

    for exp2 in experiments:
      F2 = forcings[exp2]
      w_true_2 = outputs[exp2]
      w_pred[exp2] = estimate_w(F2, operator, op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
      L2[exp2] = calc_L2(w_true_2, w_pred[exp2])
      print(f'\tTest: {exp2} - {L2[exp2]}')

  else:
    for exp1 in experiments:
      w_pred[exp1], L2[exp1] = {}, {}
      if verbose:
        print(f'Train: {exp1} - L2 Error')

      F1 = forcings[exp1]
      w_true_1 = outputs[exp1]
      operator[exp1] = create_emulator(op_type, w_true_1, F1, t=t, dt=dt, n_boxes=n_boxes, w_dict=w_dict,
                                       F_dict=F_dict, n_modes=n_modes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag,
                                       B=B, xi=xi, n_ensemble=n_ensemble, delta=delta, n_steps=n_steps)

      for exp2 in experiments:
        F2 = forcings[exp2]
        w_true_2 = outputs[exp2]
        w_pred[exp1][exp2] = estimate_w(F2, operator[exp1], op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
        L2[exp1][exp2] = calc_L2(w_true_2, w_pred[exp1][exp2])
        print(f'\tTest: {exp2} - {L2[exp1][exp2]}')

  return operator, w_pred, L2

###############################
## Ensemble Helper Functions ##
###############################

def evaluate_ensemble(experiments, n_ensemble, n_choices, forcings_ensemble, w_ensemble, op_type, op_true, w0=None, n_steps=None,
                      t=None, dt=None, n_boxes=None, w_dict=None, F_dict=None, n_modes=None, diff_flag=0, vert_diff_flag=0, B=None, xi=0, delta=0):

  operator_ensemble, operator_L2_avg, w_pred_L2 = {}, {}, {}

  # Separate treament for direct diagnosis of response function (does this even make sense in this context?)
  if op_type == 'direct':
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

  for exp_flag, exp1 in enumerate(experiments):
    forcing_w = list(zip(forcings_ensemble[exp1],w_ensemble[exp1]))
    operator_ensemble[exp1], operator_L2_avg[exp1], w_pred_L2[exp1] = [], [], {}

    # Setup data storage
    for exp2 in experiments:
      w_pred_L2[exp1][exp2] = []

    # Iterate over ensemble subsets of length n
    for n in range(1,n_ensemble + 1):
      operator_subset, operator_L2_subset, w_pred_L2_subset = [], [], {}

      for exp2 in experiments:
        w_pred_L2_subset[exp2] = []

      # Repeatedly select subsets n_choices times
      for i in range(n_choices):
        temp_choice = random.sample(forcing_w, n)
        forcing_choice, w_choice = zip(*temp_choice)

        # Take mean over the subset of ensemble members
        mean_forcing = np.mean(np.stack(forcing_choice, axis=0), axis=0)
        mean_w = np.mean(np.stack(w_choice, axis=0), axis=0)

        # Calculate operator over the subset
        if op_type == 'DMD':
          A_DMD, B_DMD = method_1a_DMD(mean_w, mean_forcing)
          operator_temp = (A_DMD, B_DMD)
        elif op_type == 'EDMD':
          A_EDMD, B_EDMD = method_1b_EDMD(mean_w, mean_forcing, w_dict, F_dict)
          operator_temp = (A_EDMD, B_EDMD)
        elif op_type == 'deconvolve':
          operator_temp = method_3a_deconvolve(mean_w, mean_forcing, dt, regularize=True)
        elif op_type == 'fit':
          operator_temp = method_4a_fit(mean_w, mean_forcing, t, dt, n_modes, n_boxes, B)
        elif op_type == 'FDT':
          operator_temp = method_2b_FDT(n, n_boxes, n_steps, xi, delta, exp_flag, diff_flag, vert_diff_flag)

        operator_subset.append(operator_temp)

        # Calculate error between ensemble and ground-truth operator
        operator_L2_subset.append(np.linalg.norm(np.array(operator_temp) - np.array(op_true[exp1])))

        # Emulate output and calculate L2 to ground truth
        for exp2 in experiments:
          forcing_true = np.mean(forcings_ensemble[exp2], axis=0)
          w_true = np.mean(w_ensemble[exp2], axis=0)

          w_pred_temp = estimate_w(forcing_true, operator_temp, op_type, dt, w0, n_steps, n_boxes, w_dict, F_dict)
          w_pred_L2_subset[exp2].append(calc_L2(w_true, w_pred_temp))

      # Calculate the average operator and error across the number of choices
      if op_type == 'DMD' or op_type == 'EDMD':
        A, B = zip(*operator_subset)
        A_mean, B_mean = np.mean(np.stack(A, axis=0), axis=0), np.mean(np.stack(B, axis=0), axis=0)
        operator_ensemble[exp1].append((A_mean, B_mean))
      elif op_type == 'deconvolve' or op_type == 'fit':
        R_mean = np.mean(np.stack(operator_subset, axis=0), axis=0)
        operator_ensemble[exp1].append((R_mean))

      operator_L2_avg[exp1].append(np.mean(operator_L2_subset))

      for exp2 in experiments:
        w_pred_L2[exp1][exp2].append(np.mean(w_pred_L2_subset[exp2]))

  return operator_ensemble, operator_L2_avg, w_pred_L2

##############################
## General Helper Functions ##
##############################
"""
def L_to_G():
  # Convert a linear operator to a response function

  return
"""

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

def plot_true_pred(T_true, T_pred, experiments):
  n_exp = len(experiments)
  fig, ax = plt.subplots(n_exp, n_exp, figsize=(12,12), constrained_layout=True)

  for i, exp1 in enumerate(experiments):
    for j, exp2 in enumerate(experiments):
      T_true_temp, T_pred_temp = T_true[exp2].T, T_pred[exp1][exp2].T
      n_boxes = T_true_temp.shape[1]
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