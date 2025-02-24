from multiprocessing.sharedctypes import Value
from venv import create
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use a LaTeX-compatible serif font
    "font.serif": ["Computer Modern Roman"],  # Or another LaTeX font
})

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

  F = np.hstack((F.T, np.zeros((F.shape[1], 2))))

  Phi_F = dict_F.transform(F[:-1,:])
  Phi_w = dict_w.transform(w[:,:-1].T)
  Phi_wprime = dict_w.transform(w[:,1:].T)

  Omega = np.concatenate([Phi_w.T,Phi_F.T])
  K = Phi_wprime.T @ np.linalg.pinv(Omega)
  n = len(Phi_wprime.T)

  A_EDMD, B_EDMD = K[:,:n], K[:,n:]

  return A_EDMD, B_EDMD

"""
def method_2a_direct():
  # Calculate G directly

  return G_direct

def method_2b_FDT():
  # Calculate G from an ensemble using the FDT

  return G_FDT

def method_3a_deconvolve(w, F):
  # Calculate G using deconvolution

  return G_deconv

def method_3b_Edeconvolve(w, F, dict_w, dict_F):
  # Calculate G using extended deconvolution

  return G_Edeconv

def method_4a_fit(w, F):
  # Calculate G using an exponential fit

  return G_fit

"""

def create_emulator(op_type, w, F, w_dict=None, F_dict=None):
  if op_type == 'DMD':
    A_DMD, B_DMD = method_1a_DMD(w, F)
    operator = (A_DMD, B_DMD)

  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = method_1b_EDMD(w, F, w_dict, F_dict)
    operator = (A_EDMD, B_EDMD)

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

def emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, dict_w, dict_F):

  # x0 is shape (3,)
  # Convert to (1,3) for dictionary.transform
  phi0 = dict_w.transform(w0.reshape(1, -1))  # shape (1, n_lifted)
  phi0 = phi0.flatten()                            # (n_lifted,)

  # Allocate
  phi_pred = np.zeros((phi0.shape[0], n_steps))
  phi_pred[:, 0] = phi0

  # For storing the reconstructed state
  w_rec = np.zeros((3, n_steps))

  # We'll fill first step
  w_rec[:, 0] = w0

  F = np.hstack((F.T, np.zeros((F.shape[1], 2))))
  Phi_F = dict_F.transform(F[:-1,:]).T

  for k in range(1, n_steps):
    # Discrete-time update in lifted space

    phi_pred[:, k] = A_EDMD @ phi_pred[:, k-1] + B_EDMD @ Phi_F[:,k-1]

    # Reconstruct the state from the first 3 dimensions
    w_rec[:, k] = phi_pred[1:4, k]  # Depends on how dictionary is defined

  return w_rec

#####################################
## Functions to Evaluate Emulators ##
#####################################

def estimate_w(w0, F, dt, operator, op_type, n_steps, dict_w=None, dict_F=None):
  # Estimate variable of interest given an initial
  # condition and forcing

  if op_type == 'DMD':
    A_DMD, B_DMD = operator
    w_est = emulate_DMD(F, A_DMD, B_DMD, w0, n_steps)
  elif op_type == 'EDMD':
    A_EDMD, B_EDMD = operator
    w_est = emulate_EDMD(F, A_EDMD, B_EDMD, w0, n_steps, dict_w, dict_F)
  else:
    raise ValueError('Operator type {op_type} not recognized.')

  return w_est

def calc_L2(w_true, w_est):
  # Estimate L2 error between emulator and ground truth
  return np.linalg.norm(w_true - w_est)

def emulate_experiments(experiments, outputs, forcings, op_type, w0, dt, n_steps, w_dict=None, F_dict=None, verbose=True):
  operator = {}
  for exp1 in experiments:
    if verbose:
      print(f'Train: {exp1} - L2 Error')

    F1 = forcings[exp1]
    w_true_1 = outputs[exp1]
    operator[exp1] = create_emulator(op_type, w_true_1, F1, w_dict, F_dict)

    for exp2 in experiments:
      F2 = forcings[exp2]
      w_true_2 = outputs[exp2]
      w_pred = estimate_w(w0, F2, dt, operator[exp1], op_type, n_steps, w_dict, F_dict)
      print(f'\tTest: {exp2} - {calc_L2(w_true_2, w_pred)}')

  return operator

##############################
## General Helper Functions ##
##############################
"""
def L_to_G():
  # Convert a linear operator to a response function

  return
"""

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