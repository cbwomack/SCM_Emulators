import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from tqdm import tqdm
import warnings
import math
import emulator_utils

colors = emulator_utils.brewer2_light
labels = ['High Lat. Ocean','Land','Low Lat. Ocean']

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use a LaTeX-compatible serif font
    "font.serif": ["Computer Modern Roman"],  # Or another LaTeX font
})

def SolverBudykoSellers(const, grid, params, init, time):

  # Get constants
  earth_radius = const['earth_radius']
  rho_w = const['rho_w']
  cp_w = const['cp_w']

  # Get grid
  lats = grid['lats']
  stag_lats = grid['stag_lats']
  NL = grid['NL']

  # Get parameters
  A = params['A']
  B = params['B']
  ASRf = params['ASRf']
  force_flag = params['force_flag']
  K = params['K']
  stag_z = params['stag_z']
  Tinit = init['T']

  # Get time
  dt = time['dt']
  NT = time['NT']
  DS = time['DS']
  NS = time['NS']

  # Check if diffusion parameters exist, otherwise set to default
  if 'initial_D' not in params:
    initial_D = params['D']
  else:
    initial_D = params['initial_D']
  if 'final_D' not in params:
    final_D = params['D']
  else:
    final_D = params['final_D']
  if 'final_T' not in params:
    final_T = 1
  else:
    final_T = params['final_T']
  if 'reg_D' not in params:
    reg_D = []
  else:
    reg_D = params['reg_D']

  # Check if noise parameter exists, otherwise ignore it
  if 'xi' not in params:
    xi = 0
  else:
    xi = params['xi']

  # Check if perturbation exists, otherwise ignore it
  if 'delta' not in params:
    delta = 0
  else:
    delta = params['delta']

  if 'noise_ts' in params:
    noise_ts = params['noise_ts']

  avg_D = (final_D + initial_D)/2
  diff_D2 = (final_D - initial_D)/2

  # Initialize solver
  out = {}
  NZ = stag_z.shape[1] - 1
  dzs = stag_z[:, 1:] - stag_z[:, :-1]
  gamma = K
  C = dzs * rho_w * cp_w
  S = np.zeros_like(lats)

  # Latitude-related calculations
  coslat = np.cos(lats * np.pi / 180)
  dlat = np.diff(stag_lats)
  cos_factor = coslat / np.mean(coslat)
  weight_factor = (cos_factor * dlat) / np.sum(cos_factor * dlat)

  # Switch the ASR function
  if ASRf == 0:
    # ASR is not calculated at all -- Perturbation equation with OLR giving the feedback (S=0)
    ASR_func = lambda TAS: 0

  elif ASRf == 1:
    # ASR is calculated averaging daily insolation
    # Shortwave feedback is only sea-ice (discontinuous)
    #for i in range(NL):
    #  S[i] = np.mean(daily_insolation(0, lats[i], np.arange(1, 365.24, 0.1)))

    a0 = params['a0']
    a2 = params['a2']
    ai = params['ai']
    Tf = params['Tf']

    # Compute non-frozen albedo
    alphanf = a0 + a2 * legendre(2)(np.sin(lats * np.pi / 180))
    alphaf = ai

    ASR_func = lambda TAS: S * (1 - (alphanf * (TAS > Tf) + alphaf * (TAS <= Tf)))

  elif ASRf == 2:
    # ASR is calculated with a linear function; shortwave feedbacks included
    ASW = params['ASW']
    lambdaSW = params['lambdaSW']
    ASR_func = lambda TAS: ASW + lambdaSW * TAS

  elif ASRf == 3:
    # ASR is calculated with varying albedo only (perturbation)
    ao = params['ao']
    ai = params['ai']
    Thalf = params['Thalf']
    tscale = params['tscale']
    idx_af = params['idx_af']
    Sparam = params['S']

    # Set S different from zero only where I want the feedback activated
    for i in idx_af:
      S[i] = Sparam

    ainitial = 0.5 * (ao + ai) + 0.5 * (ao - ai) * np.tanh((0 - Thalf) / tscale)
    ASR_func = lambda TAS: S * (ainitial - (0.5 * (ao + ai) + 0.5 * (ao - ai) * np.tanh((TAS - Thalf) / tscale)))

  # Switch the forcing function for both shortwave and longwave
  if force_flag == 0:
    # Abrupt
    reff_sw = params['reff_sw']
    reff_lw = params['reff_lw']
    RFSW = lambda t: reff_sw
    RFLW = lambda t: reff_lw

  elif force_flag == 1:
    # High Emissions
    RF_end = params['RF_end']  # W m-2
    RF_init = params['RF_init']  # W m-2
    t_star = params['t_star']  # years
    t_final = time['int_yrs']  # years
    amp = (RF_end - RF_init) / np.exp(t_final / t_star)

    RFSW = lambda t: 0
    RFLW = lambda t: amp * np.exp(t / t_star)

  elif force_flag == 2:
    # Mid. Emissions
    beta = 1/50
    RFSW = lambda t: 0
    RFLW = lambda t: 2.25 + 2.25*np.tanh(beta*(t - 150))/np.tanh(250*beta)

  elif force_flag == 3:
    # Overshoot
    a = params['a'] # [W m-2]
    b = params['b'] # [years]
    c = params['c'] # [growth rate]

    RFSW = lambda t: 0
    RFLW = lambda t: a * np.exp(-np.power(t - b,2)/(2 * c**2))

  elif force_flag == 4:
    # Impulse forcing case (longwave only)
    RFSW = lambda t: 0
    RFLW = lambda t: 1 if t <= 1 else 0

  ## Time integration
  # Initialize main variables
  T_new = Tinit
  #if 'delta' in params:
  #  T_new += delta

  # Create NaN-filled arrays with the appropriate shapes
  T_ts = np.full((NL, NZ, NS), np.nan)
  ASR_ts = np.full((NL, NS), np.nan)
  OLR_ts = np.full((NL, NS), np.nan)
  HDYN_ts = np.full((NL, NS), np.nan)
  #VDYN_ts = np.full((NL, NS), np.nan)
  VDYN_ts = np.full((NL, NZ, NS), np.nan)
  TANM_ts = np.full((1, NS), np.nan)
  D_ts = np.full((NL + 1, NS), np.nan)
  forcing_ts = np.full((1, NS), np.nan)

  if 'noise_ts' not in params:
    noise_ts = np.full((1, NS), np.nan)

  idxsave = 0
  stag_D = np.ones(NL + 1) * initial_D

  # Compute initial global average
  gl_avg_ini = np.sum(Tinit[:, 0] * weight_factor)

  tc_years = 0  # Current time in years

  # Initialize heat fluxes
  hhfx = np.zeros(NL + 1)
  vhfx = np.zeros((NL, NZ + 1))

  # Set boundary conditions for heat fluxes
  hhfx[0] = 0
  hhfx[-1] = 0
  vhfx[:, 0] = 0
  vhfx[:, -1] = 0

  HDYN = np.full(lats.shape, np.nan)
  VDYN = np.zeros((NL, NZ))

  # Main integration loop
  for n in tqdm(range(1,NT), disable=True):
    # Set the current temperature
    T_arr = T_new

    if n == 1 and 'noise_ts' not in params:
      noise = np.random.normal(loc=0.0, scale=xi)
    elif n == 1 and 'noise_ts' in params:
      noise = noise_ts[0,idxsave]

    ######################################
    ## 1. Calculate terms for first box ##
    ######################################

    # Calculate ASR
    ASR = ASR_func(T_arr[:,0].T) - RFSW(tc_years)

    # Calculate OLR
    if tc_years <= 1:
      RFLW_n = RFLW(tc_years) + noise + delta
    else:
      RFLW_n = RFLW(tc_years) + noise

    OLR = (A - RFLW_n + B*T_arr[:,0].T)
    if 'spatial' in params:
      OLR[params['spatial'][0]] = 0
      OLR[params['spatial'][1]] = 0

    # Calculate stag_D depending on current temperatures
    gl_avg = np.sum(T_arr[:,0].T*weight_factor)
    dgl_avg = gl_avg - gl_avg_ini
    stag_D[reg_D] = avg_D - diff_D2 * np.tanh(-1/(final_T * 0.5) * (dgl_avg - final_T/2))

    # Calculate staggered horizontal heat flux
    for i in range(1, NL):
      hhfx[i] = stag_D[i] * np.cos(stag_lats[i] * np.pi/180) * (T_arr[i,0] - T_arr[i - 1,0]) / ((lats[i] - lats[i - 1]) * np.pi/180)

    # Calculate dynamic heating rate
    for i in range(0, NL):
      HDYN[i] = 1/np.cos(lats[i] * np.pi/180) * (hhfx[i + 1] - hhfx[i]) / ((stag_lats[i + 1] - stag_lats[i]) * np.pi/180)

    # Calculate vertical heat flux
    for k in range(1, NZ):
      for i in range(0, NL):
        vhfx[i,k] = -gamma * (T_arr[i,k] - T_arr[i, k - 1])
    VDYN[:,0] = vhfx[:,0] - vhfx[:,1]

    # Advance temperature
    T_new[:,0] = T_arr[:,0] + dt * np.divide(ASR - OLR.T + HDYN.T + VDYN[:,0], C[:,0])

    ###########################
    ## 2. Diffuse vertically ##
    ###########################

    for k in range(1, NZ):
      VDYN[:,k] = (vhfx[:,k] - vhfx[:,k + 1])
      T_new[:,k] = T_arr[:,k] + dt * np.divide(VDYN[:,k], C[:,k])

    ################################
    ## 3. Save off the timeseries ##
    ################################

    if n % DS == 0 or n == 1:
      # Save timeseries
      T_ts[:,:,idxsave] = T_arr
      ASR_ts[:,idxsave] = ASR
      OLR_ts[:,idxsave] = OLR
      HDYN_ts[:,idxsave] = HDYN
      D_ts[:,idxsave] = stag_D.T
      VDYN_ts[:,:,idxsave] = VDYN[0]
      TANM_ts[0,idxsave] = dgl_avg
      forcing_ts[:,idxsave] = RFLW_n
      if 'noise_ts' not in params:
        noise_ts[0,idxsave] = noise
        noise = np.random.normal(loc=0.0, scale=xi)
      else:
        noise = noise_ts[0,idxsave]
      idxsave += 1

    # Update the clock
    tc_years = tc_years + dt/365/24/3600

  # Prepare out dict
  out['H'] = (-2 * np.pi * earth_radius**2 * hhfx) / 1e15 # [PW]
  out['vhfx'] = vhfx
  out['eltime'] = tc_years
  out['T'] = T_new
  out['ASR'] = ASR
  out['OLR'] = OLR
  out['HDYN'] = HDYN
  out['VDYN'] = VDYN
  out['Savg'] = np.sum(S * weight_factor)
  out['T_ts'] = T_ts
  out['S'] = S
  out['diseq'] = np.mean(abs(ASR - OLR + HDYN))
  out['wf'] = weight_factor
  out['ASR_ts'] = ASR_ts
  out['OLR_ts'] = OLR_ts
  out['HDYN_ts'] = HDYN_ts
  out['VDYN_ts'] = VDYN_ts
  out['T_avg'] = np.sum(out['T'].T * out['wf'])
  out['stag_D'] = stag_D
  out['t_save'] = time['tvec_save']
  out['D_ts'] = D_ts
  out['TANM_ts'] = TANM_ts
  out['stag_z'] = stag_z
  out['NZ'] = NZ
  out['forcing_ts'] = forcing_ts
  out['noise_ts'] = noise_ts

  if out['diseq'] > 1e-3:
    warnings.warn('Simulation has not reached equilibirum (diseq > 1e-3 W m-2)')

  return out

def Run_Budyko_Sellers(scen_flag=0, diff_flag=0, vert_diff_flag=0, xi=0, delta=0, n_boxes=3, noise_ts=None, spatial_flag=0, F0=3.7, int_yrs=250):
  # Initialize dictionaries
  grid, params, init, const, time = {}, {}, {}, {}, {}

  # Set Constants
  const['earth_radius'] = 6.373e6 # Earth's radius [m]
  const['rho_w'] = 1e3 # Water density [kg m-3]
  const['cp_w'] = 4e3 # Water specific heat capacity [J kg-1 K-1]

  # Set grid parameters
  if vert_diff_flag == 0: # Regular grid
    if n_boxes == 5:
      grid['stag_lats'] = np.arange(-90, 91, 36)              # Vector of staggered latitudes [degrees]
      grid['dz_slabs'] = np.array([1500, 10, 150, 10, 1500])  # Water slabs thickness [m]
    elif n_boxes == 3:
      grid['stag_lats'] = np.arange(-90, 91, 60)              # Vector of staggered latitudes [degrees]
      grid['dz_slabs'] = np.array([1500, 10, 150])  # Water slabs thickness [m]
  elif vert_diff_flag == 1: # Grid with vertical diffusion
    grid['stag_lats'] = np.array([-90,90])
    grid['dz_slabs'] = np.array([])
  grid['lats'] = 0.5 * (grid['stag_lats'][1:] + grid['stag_lats'][0:-1])          # Unstaggered latitudes
  grid['NL'] = len(grid['stag_lats']) - 1                                         # Number of regions

  # Set ODE parameters
  if vert_diff_flag == 0: # Regular grid
    params['ASRf'] = 0                                      # Select shortwave radiation scheme [Daily insolation/No SW/linear]
    params['A'] = 0                                         # Intercept in OLR calculation, used for climatology [W m-2]
    if n_boxes == 5:
      params['B'] = np.array([0.67, 0.86, 2.0, 0.86, 0.67])   # Feedback parameters [W m-2 K-1]
    elif n_boxes == 3:
      params['B'] = np.array([0.67, 0.86, 2.0])   # Feedback parameters [W m-2 K-1]
    params['K'] = 0                                         # Vertical diffusivity [W m-2 K-1]
    params['stag_z'] = np.hstack([                                                  # Water box thicknesses
        np.zeros((len(grid['dz_slabs']), 1)),
        np.array(grid['dz_slabs']).reshape(-1, 1)
    ])

  elif vert_diff_flag == 1: # Grid with vertical diffusion
    params['ASRf'] = 0                                      # Select shortwave radiation scheme [Daily insolation/No SW/linear]
    params['A'] = 0                                         # Intercept in OLR calculation, used for climatology [W m-2]
    params['B'] = 0.86   # Feedback parameters [W m-2 K-1]
    params['K'] = 0.7                                        # Vertical diffusivity [W m-2 K-1]
    params['stag_z'] = np.array([[0.0, 55.2105, 709.7600]])

  if spatial_flag != 0:
    grid['dz_slabs'] = np.array([10, 150, 1500])  # Water slabs thickness [m]
    params['B'] = np.array([0.86, 2.0, 0.67])   # Feedback parameters [W m-2 K-1]

    if spatial_flag == 1:
      params['spatial'] = (1,2)
    elif spatial_flag == 2:
      params['spatial'] = (0,2)
    elif spatial_flag == 3:
      params['spatial'] = (0,1)

  # Initialize temperature
  init['T0'] = 0                                          # Initial temperature [K]
  if vert_diff_flag == 0:
    init['T'] = np.ones((grid['NL'], 1)) * init['T0']                               # Initial temperature
  elif vert_diff_flag == 1:
    init['T'] = np.array([[0.,0.]])

  # Set time parameters
  time['int_yrs'] = int_yrs                                                            # Integration time [years]
  time['save_f'] = 365                                                            # Save every time.save_f [days]
  time['dt'] = 3600 * 24 * 7                                                           # Time step [seconds]
  time['NT'] = int(round((time['int_yrs'] * 3600 * 24 * 365.24) / time['dt']))    # Number of time steps
  time['DS'] = int(round(time['save_f'] * 3600 * 24 / time['dt']))                # Save every time['DS']
  time['NS'] = math.floor(time['NT'] / time['DS']) + 1                            # Number of save times
  time['tvec_save'] = np.arange(0, time['save_f'] * time['NS'], time['save_f'])   # Vector of saved time steps

  # Set scenario parameters
  ## Uncoupled
  if diff_flag == 0:
    params['D']= 0 # Horizontal diffusivity [W m-2 K-1]

  ## Coupled
  elif diff_flag == 1:
    params['D'] = 0.55 # Horizontal diffusivity [W m-2 K-1]

  ## Abrupt 2xCO2
  if scen_flag == 0:
    params['force_flag']  = 0                                   # Select which type of forcing
    params['reff_lw']     = F0                                 # Longwave forcing [W m-2]
    params['reff_sw']     = 0                                   # Shortwave forcing [W m-2]

  ## High emissions
  elif scen_flag == 1:
    params['force_flag']  = 1
    params['RF_end']      = 8.5 # [W m-2]
    params['RF_init']     = 0.0 # [W m-2]
    params['t_star']      = 50  # [years]

  ## Mid. emissions
  elif scen_flag == 2:
    params['force_flag']  = 2
    params['RF_end']      = 4.5 # [W m-2]
    params['RF_init']     = 0.0 # [W m-2]
    params['t_star']      = 50  # [years]

  ## Overshoot
  elif scen_flag == 3:
    params['force_flag']  = 3
    params['a']           = 4     # [W m-2]
    params['b']           = 200   # [years]
    params['c']           = 42.47 # [growth rate]
    #params['c']           = 60 # [growth rate]

  ## Impulse
  elif scen_flag == 4:
    params['force_flag']  = 4

  else:
    raise ValueError(f'Error, scenario {scen_flag} not recognized.')

  if xi != 0:
    params['xi'] = xi

  if delta != 0:
    params['delta'] = delta

  if noise_ts is not None:
    params['noise_ts'] = noise_ts

  return SolverBudykoSellers(const, grid, params, init, time)

def plot_BudykoSellers(out):

  fig, ax = plt.subplots(constrained_layout=True)
  for i in range(len(out['T_ts'])):
    ax.plot(np.squeeze(out['T_ts']).T[:,i], c=colors(i), lw=2)#, label=labels[i], lw=2)

  ax.set_xlabel('Year')
  ax.set_ylabel('Temperature')
  ax.legend()

  return





