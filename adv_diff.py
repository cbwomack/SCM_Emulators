import numpy as np
import matplotlib.pyplot as plt

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