# Required imports
## Basic imports
import numpy as np

## Other
from utils_emulator import open_results
from scipy.stats import gaussian_kde

## Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import seaborn as sns
from cmcrameri import cm

## Setup plots
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica Light"],
})

##############
## Figure 2 ##
##############

def plot_baseline_fig2(scen, val_mean, val_std, t_vec, only_mean=False, save_fig=False, fig_name=None):
  """
  Plot ensemble-mean baseline temperature (and ±1 std band) for a
  single scenario.

  Parameters
  ----------
  scen : str
    Scenario key to select data from `val_mean` and `val_std`.
  val_mean, val_std : dict
    Dictionaries holding mean and standard-deviation arrays
    indexed by scenario.
  t_vec : ndarray
    Time vector for the x-axis.
  only_mean : bool, optional
    If True, suppress shaded ±1 std envelopes.
  save_fig : bool, optional
    Save the figure to PDF when True.
  fig_name : str, optional
    Filename (without extension) used when `save_fig` is True.

  Returns
  -------
  None
  """

  # Figure setup
  fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

  # Plot each component and standard deviation (optional)
  for i in range(len(val_mean[scen])):
    ax.plot(t_vec, val_mean[scen][i], c=cm.batlowS(i + 3), lw=2)
    if not only_mean:
      ax.fill_between(t_vec, val_mean[scen][i] - val_std[scen][i], val_mean[scen][i] + val_std[scen][i], alpha=0.5, color=cm.batlowS(i + 3))

  # Axis labels and style
  ax.set_ylabel(r'Temperature [$^\circ$C]',fontsize=18)
  ax.set_xlabel('Year',fontsize=18)
  ax.tick_params(axis='both', which='major', labelsize=14)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

def plot_alt_fig2(scenarios, val_mean, val_std, t_vec, save_fig=False, fig_name=None):
  """
  Compare ensemble mean temperature trajectories (±1 std) across
  multiple scenarios.

  Parameters
  ----------
  scenarios : list
    Ordered scenario names to display.
  val_mean, val_std : dict
    Mean and standard deviation arrays keyed by scenario.
  t_vec : ndarray
    Time vector for the x-axis.
  save_fig : bool, optional
    Save figure as PDF when True.
  fig_name : str, optional
    Filename (without extension) used if `save_fig` is True.

  Returns
  -------
  None
  """

  # Figure setup
  n_scens = len(scenarios)
  fig, ax = plt.subplots(1, n_scens, figsize=(6*n_scens,4), constrained_layout=True, sharey=True)

  # Plot each component and standard deviation
  for i, scen in enumerate(scenarios):
    for j in range(len(val_mean[scen])):
      ax[i].plot(t_vec, val_mean[scen][j], c=cm.batlowS(j + 3), lw=2)
      ax[i].fill_between(t_vec, val_mean[scen][j] - val_std[scen][j], val_mean[scen][j] + val_std[scen][j], alpha=0.5, color=cm.batlowS(j + 3))

    ax[i].set_xlabel('Year',fontsize=18)
    ax[i].tick_params(axis='both', which='major', labelsize=14)

  ax[0].set_ylabel(r'Temperature [$^\circ$C]',fontsize=18)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

def plot_pdf_fig2(scen, ensemble, save_fig=False, fig_name=None):
  """
  Plot probability densities of ensemble anomalies for each region within a scenario.

  Parameters
  ----------
  scen : str
    Scenario key in `ensemble`.
  ensemble : dict
    Scenario -> ndarray (n_ensemble, n_region, n_time).
  save_fig : bool, optional
    Save plot when True.
  fig_name : str, optional
    Filename (without extension) for saving.

  Returns
  -------
  None
  """

  # Figure setup
  fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
  n_pts = 300
  data = np.array(ensemble[scen])

  mean_state = data.mean(axis=0, keepdims=True)
  anomalies  = data - mean_state
  n_ensemble, n_region, n_time = data.shape
  bandwidth='scott'

  # Calculate and plot PDF
  for i in range(n_region):
    flat = anomalies[:, i, :].ravel()
    kde  = gaussian_kde(flat, bw_method=bandwidth)

    xs = np.linspace(flat.min(), flat.max(), n_pts)
    ax.plot(xs, kde(xs), c=cm.batlowS(i + 3), lw=2)

  ax.set_xlim([-1,1])
  ax.set_xlabel("Value",fontsize=18)
  ax.set_ylabel("Temperature PDF",fontsize=18)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return


##############
## Figure 3 ##
##############

def plot_scenarios(experiments, scenarios, val_mean, val_std, t_vec_box, t_vec_lorenz, regions, baseline_mean, save_fig=False, fig_name=None):
  """
    Visualise mean temperature (or Lorenz Z) evolutions for multiple
    experiment-scenario combinations in a grid of subplots.

    Parameters
    ----------
    experiments : list
      Names of experiment groups (rows).
    scenarios : list
      Scenario labels (columns).
    val_mean, val_std : dict
      Nested dicts holding mean and ± std data keyed as
      val_mean[experiment][scenario].
    t_vec_box, t_vec_lorenz : ndarray
      Time vectors for box model plots and Lorenz plots, respectively.
    regions : list
      Region names used in legends.
    baseline_mean : ndarray
      Baseline Lorenz mean for anomaly plotting.
    save_fig : bool, optional
      Save figure as PDF when True.
    fig_name : str, optional
      Filename stem used if `save_fig` is True.

    Returns
    -------
    None
    """

  # Figure setup
  n_exp, n_scen, n_boxes = len(experiments), len(scenarios), len(regions)
  fig, ax = plt.subplots(n_exp, n_scen, figsize=(4*n_scen,4*n_exp), constrained_layout=True, sharex='col', sharey='row')

  # Iterate over experiments
  for i, exp in enumerate(experiments):
    val_mean_temp = val_mean[exp]
    if i == 1:
      regions = ['Atmosphere','Ocean']

    # Iterate over scenarios
    for j, scen in enumerate(scenarios):
      val_mean_plot = val_mean_temp[scen].T
      # Box model
      if i != 2:
        n_boxes = val_mean_plot.shape[1]
        for k in range(n_boxes):
            ax[i,j].plot(t_vec_box, val_mean_plot[:,k], lw=3.5, c=cm.batlowS(k + 3), label = regions[k])
      # Cubic Lorenz
      else:
        ax[i,j].plot(t_vec_lorenz, val_mean_plot - baseline_mean[2], color=cm.batlowS(3), lw=2.5, label='Mean')
        ax[i,j].fill_between(t_vec_lorenz, val_mean_plot - val_std[scen] - baseline_mean[2], val_mean_plot + val_std[scen] - baseline_mean[2], alpha=0.4, color=cm.batlowS(3), label=r'$\sigma$')

      if j == 0 and i != 2:
        ax[i,j].legend(fontsize=20)
        ax[i,j].set_ylabel(r'Temperature [$^\circ$C]',fontsize=22)
      elif j == 0 and i == 2:
        ax[i,j].set_ylabel(r'$\langle Z \rangle$',fontsize=22)

      if i == n_exp - 1:
        ax[i,j].set_xlabel('Year',fontsize=24)

      # Add scenario names as titles
      if i == 0:
        if scen == 'Abrupt':
          title = r'\textit{Abrupt}'
        elif scen == 'High Emissions':
          title = r'\textit{High Emissions}'
        elif scen == 'Mid. Emissions':
          title = r'\textit{Mid. Emissions}'
        elif scen == 'Overshoot':
          title = r'\textit{Overshoot}'
        ax[i,j].set_title(title, fontsize=24, va="center")

      ax[i,j].tick_params(axis='both', which='major', labelsize=18)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

##############
## Figure 4 ##
##############

def get_error_by_method(scenarios, ensemble_index):
  """
  Collect median emulator errors for each method over four experiments.

  Parameters
  ----------
  scenarios : list
    Scenario names present in the saved error dictionaries.
  ensemble_index : int
    Ensemble member to extract for experiments 3-4.

  Returns
  -------
  dict
    {'PS', 'FDT', 'Deconv.', 'Modal', 'DMD', 'EDMD'} →
    list of median errors for experiments 1-4.
  """

  # Initialize output container
  error_by_method = {}

  # Loop over methods and experiments
  for method in ['I_PS','II_FDT','III_deconv','IV_modal','V_DMD','VI_EDMD']:
    error_by_method[method] = []
    for exp in ['1','2','3','4']:
      # Load error data
      if exp == '3' or exp == '4':
        name = f'exp{exp}_{method}_error_ensemble'
      else:
        name = f'exp{exp}_{method}_error'
      error_dict = open_results(name)

      # Make list of the errors (exclude train-test same)
      error_list_temp = []
      for train in scenarios:
        if method == 'II_FDT':
          if exp == '3' or exp == '4':
            error_list_temp.append(np.mean(error_dict[train][ensemble_index]))
          else:
            error_list_temp.append(np.mean(error_dict[train]))
          continue
        for test in scenarios:
          if train == test:
            continue
          if exp == '3' or exp == '4':
            error_list_temp.append(np.mean(error_dict[train][test][ensemble_index]))
          else:
            error_list_temp.append(np.mean(error_dict[train][test]))

      error_by_method[method].append(np.median(error_list_temp))

  # DMD and EDMD data are the same for these experiments,
  # any differences are from regularization
  error_by_method['VI_EDMD'][0] = error_by_method['V_DMD'][0]
  error_by_method['VI_EDMD'][2] = error_by_method['V_DMD'][2]

  # Format output
  all_errors = {
    'PS': (error_by_method['I_PS']),
    'FDT': (error_by_method['II_FDT']),
    'Deconv.': (error_by_method['III_deconv']),
    'Modal': (error_by_method['IV_modal']),
    'DMD': (error_by_method['V_DMD']),
    'EDMD': (error_by_method['VI_EDMD']),
  }

  return all_errors

def plot_emulator_bars(experiments, all_errors, save_fig=False, fig_name=None):
  """
  Draw clustered bar charts of median NRMSE for several emulator
  methods across multiple experiments.

  Parameters
  ----------
  experiments : list
    Ordered names of the four experiments.
  all_errors : dict
    Method -> list of median NRMSE values (one per experiment).
  save_fig : bool, optional
    Save the figure to *Figures/<fig_name>.pdf* when True.
  fig_name : str, optional
    Filename stem used when `save_fig` is True.

  Returns
  -------
  None
  """

  # Hatch styles by method
  hatch_map = {
    'PS': '',
    'FDT': '/',
    'Deconv.': '/',
    'Modal': '/',
    'DMD': 'x',
    'EDMD': 'x'
  }

  # Setup plots
  cluster_spacing = 0.8
  x = np.arange(len(experiments)) * cluster_spacing
  width = 0.1

  # Create two subplots, sharing the x-axis
  fig, ax = plt.subplots(sharex=True, layout='constrained')
  fig.set_size_inches(12, 5)

  # Plot Bars and Annotations on Both Subplots ---
  def plot_bars(ax):
    multiplier = 0
    colors = iter(cm.batlowS.colors)

    for attribute, measurement in all_errors.items():
      offset = width * multiplier
      color = next(colors)
      hatch_style = hatch_map.get(attribute, '') # Look up hatch style
      rects = ax.bar(x + offset, measurement, width, label=attribute, color=color,
                  hatch=hatch_style, edgecolor='black', linewidth=0.6)

      # Add bar labels with 2 significant digits and matching color
      for rect in rects:
        height = rect.get_height()
        # Check for 'nan' data to label as 'N/a'
        if np.isnan(height):
          label_text = 'N/a'
          y_pos = 0
        else:
          if height < 1:
            height = round(height, 2)
          label_text = f'{height:.3g}'
          y_pos = height

        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color,
                    fontsize=12)
      multiplier += 1

  # Plot the data on both the top and bottom axes
  plot_bars(ax)
  ax.set_ylim(0, 35)
  ax.tick_params(axis='x', which='both', bottom=False)
  ax.set_title('Emulator performance by experiment', fontsize=22)

  hatch_handles = [
    Patch(facecolor=cm.batlowS(0), edgecolor='k', lw=0.5, label='PS'),
    Patch(facecolor=cm.batlowS(1), edgecolor='k', lw=0.5, label='FDT'),
    Patch(facecolor=cm.batlowS(2), edgecolor='k', lw=0.5, label='Deconv.'),
    Patch(facecolor=cm.batlowS(3), edgecolor='k', lw=0.5, label='Modal'),
    Patch(facecolor=cm.batlowS(4), edgecolor='k', lw=0.5, label='DMD'),
    Patch(facecolor=cm.batlowS(5), edgecolor='k', lw=0.5, label='EDMD'),
    Patch(facecolor='white', edgecolor='black', hatch='/', label='Response Fcns.'),
    Patch(facecolor='white', edgecolor='black', hatch='x', label='Operators')
  ]

  ax.legend(handles=hatch_handles, loc='upper left', ncols=3, fontsize=14)
  fig.supylabel(r'NRMSE [\%]', fontsize=20)
  num_methods = len(all_errors)
  tick_pos = x + width * (num_methods - 1) / 2
  ax.set_xticks(tick_pos)
  ax.set_xticklabels(experiments)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

###################
## Figures 5 & 7 ##
###################

def plot_single_heatmap(error_metrics: dict,
                        method: str,
                        train_scenarios: list[str],
                        test_scenarios: list[str],
                        ax: plt.Axes,
                        vmax: float,
                        cmap: str = "Reds",
                        long_title: str = '',
                        add_xlabel: bool = True,
                        add_ylabel: bool = True,
                        add_cbar: bool = False) -> None:
  """
  Draw one NRMSE heat-map for a given emulator method.

  Parameters
  ----------
  error_metrics : dict
    Nested errors indexed as error_metrics[method][train][test].
  method : str
    Key selecting the emulator within `error_metrics`.
  train_scenarios, test_scenarios : list[str]
    Scenario order for rows (train) and columns (test).
  ax : matplotlib.axes.Axes
    Target axis for the heat-map.
  vmax : float
    Colour-scale maximum (min is fixed at 0).
  cmap : str, optional
    Matplotlib/SNS colour-map (default "Reds").
  long_title : str, optional
    Sub-plot title.
  add_xlabel, add_ylabel : bool, optional
    Toggle axis tick-labels.
  add_cbar : bool, optional
    Add colour-bar on this axis when True.

  Returns
  -------
  None
  """

  # Instantiate data array
  data = np.empty((len(train_scenarios), len(test_scenarios)))
  for i, scen_train in enumerate(train_scenarios):
    for j, scen_test in enumerate(test_scenarios):
      if scen_train == scen_test:
        data[j, i] = np.nan
        continue

      try:
        value = np.mean(error_metrics[method][scen_train][scen_test])
      except KeyError:
        value = np.nan
      data[j, i] = value

  # Plot the heatmap using the provided axis and vmax
  sns.heatmap(
    data,
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    linewidth=0.5,
    annot=True,
    fmt=".2g",
    cbar=add_cbar,
    cbar_kws={"label": r"NRMSE [\%]"} if add_cbar else None
  )

  # Configure labels and title for the subplot
  ax.set_title(long_title)
  tick_labels = ['Abr.','Hi. Em.', 'Mid. Em.', 'Over.']

  if add_xlabel:
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
  else:
    # Hide x-axis labels if not needed (for top row plots)
    ax.set_xlabel("")
    ax.set_xticklabels([])

  if add_ylabel:
    ax.set_yticklabels(tick_labels, rotation=45)
  else:
    # Hide y-axis labels if not needed (for right column plots)
    ax.set_ylabel("")
    ax.set_yticklabels([])

  return

def plot_error_heatmap_grid(error_metrics: dict,
                            methods: list[str],
                            titles: list[str],
                            train_scenarios: list[str],
                            test_scenarios: list[str],
                            cmap: str = "Reds",
                            save_fig: bool = False,
                            fig_name: str = "None"):
  """
  Display a 2 x 2 grid of NRMSE heat-maps comparing emulator skill
  across multiple methods and scenario pairs.

  Parameters
  ----------
  error_metrics : dict
    Nested structure returned by `get_error_by_method`.
  methods : list[str]
    Four method keys to plot, ordered row-major.
  titles : list[str]
    Sub-plot titles corresponding to `methods`.
  train_scenarios, test_scenarios : list[str]
    Scenario order for rows (train) and columns (test).
  cmap : str, optional
    Colour-map passed to seaborn (default "Reds").
  save_fig : bool, optional
    Write PDF to *Figures/<fig_name>.pdf* when True.
  fig_name : str, optional
    Filename stem used if `save_fig` is True.

  Returns
  -------
  None
  """

  # Figure setup
  fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex='col')
  panel_labels = ['(a)', '(b)', '(c)', '(d)']
  global_vmax = 10

  mappable = None
  for i, ax in enumerate(axes.flat):
    add_xlabel = (i >= 2)
    add_ylabel = (i % 2 == 0)

    plot_single_heatmap(
      error_metrics=error_metrics,
      method=methods[i],
      train_scenarios=train_scenarios,
      test_scenarios=test_scenarios,
      ax=ax,
      vmax=global_vmax,
      cmap=cmap,
      long_title=titles[i],
      add_xlabel=add_xlabel,
      add_ylabel=add_ylabel
    )
    if i == 0:
      mappable = ax.collections[0]

    ax.text(-0.09, 1.08, panel_labels[i], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

  # Figure formatting
  fig.supxlabel("Train scenario", fontsize=24)
  fig.supylabel("Test scenario", fontsize=24)
  fig.tight_layout()
  fig.subplots_adjust(right=0.85)
  cbar_ax = fig.add_axes([0.88, 0.17, 0.03, 0.778])
  cbar = fig.colorbar(mappable, cax=cbar_ax)
  cbar.set_label(r"NRMSE [\%]", size=14)
  cbar.outline.set_visible(False)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf', dpi=300)

  return

##############
## Figure 6 ##
##############

def plot_box_true_pred(experiments, scenarios, T_out, T_pred, regions, save_fig=False, fig_name=None):
  """
  Plot true vs. emulated box-model temperatures for multiple
  experiments (rows) and scenarios (columns).

  Parameters
  ----------
  experiments : list
    Experiment labels for row ordering.
  scenarios : list
    Scenario labels for column ordering.
  T_out : dict
    Nested true outputs: T_out[experiment][scenario] -> array (n_box, n_time).
  T_pred : dict
    Emulator predictions keyed by scenario, same shape as T_out entries.
  regions : list
    Names of the model boxes for legends.
  save_fig : bool, optional
    Save the figure as *Figures/<fig_name>.pdf*.
  fig_name : str, optional
    Output filename stem when `save_fig` is True.

  Returns
  -------
  None
  """

  # Figure setup
  n_exp, n_scen, n_boxes = len(experiments), len(scenarios), len(regions)
  fig, ax = plt.subplots(n_exp, n_scen, figsize=(4*n_scen,5*n_exp), constrained_layout=True, sharex='col', sharey='row')

  # Custom handles
  truth_handle = mlines.Line2D([], [], color='k', linestyle='-',  label='Truth', lw=3.5, alpha=0.5)
  emu_handle   = mlines.Line2D([], [], color='k', linestyle='-.', label='Emulator', lw=3.5)

  nice_handle1 = mlines.Line2D([], [], color=cm.batlowS(3), linestyle='-',  label='High Lat. Ocean', lw=3.5)
  nice_handle2 = mlines.Line2D([], [], color=cm.batlowS(4), linestyle='-',  label='Land', lw=3.5)
  nice_handle3 = mlines.Line2D([], [], color=cm.batlowS(5), linestyle='-',  label='Low Lat. Ocean', lw=3.5)

  # Loop over experiments and scenarios
  for i, exp in enumerate(experiments):
    T_out_temp = T_out[exp]

    for j, scen in enumerate(scenarios):
      T_true_temp = T_out_temp[scen].T
      T_pred_temp = T_pred[scen].T
      n_boxes = T_true_temp.shape[1]
      for k in range(n_boxes):
        ax[j].plot(T_true_temp[:,k], lw=3, c=cm.batlowS(k + 3), label = regions[k], alpha=0.5)
        ax[j].plot(T_pred_temp[:,k], lw=3, c=cm.batlowS(k + 3), ls='-.')

      if j == 0:
        ax[j].legend([nice_handle1, nice_handle2, nice_handle3], ['High Lat. Ocean', 'Land', 'Low Lat. Ocean'], fontsize=20, loc='upper left')
        ax[j].set_ylabel(r'Temperature [$^\circ$C]',fontsize=22)

      if j == 1:
        ax[j].legend([truth_handle, emu_handle],['Truth', 'Emulator'], fontsize=20)

      if i == n_exp - 1:
        ax[j].set_xlabel('Year',fontsize=24)

      if i == 0:
        if scen == 'Abrupt':
          title = r'\textit{Abrupt}'
        elif scen == 'High Emissions':
          title = r'\textit{High Emissions}'
        elif scen == 'Mid. Emissions':
          title = r'\textit{Mid. Emissions}'
        elif scen == 'Overshoot':
          title = r'\textit{Overshoot}'
        ax[j].set_title(title, fontsize=18, va="center")

      ax[j].tick_params(axis='both', which='major', labelsize=18)

  fig.suptitle('Emulator performance, Method II: FDT', fontsize=24)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

###################
## Figures 8 & 9 ##
###################

def plot_ensemble_error_multi(NRMSE_all, NRMSE_all_base, scenarios, methods, train, cubLor=False, save_fig=False, fig_name=None):
  """
  Plot how emulator error (NRMSE) changes with ensemble size for several
  methods and scenarios.

  Parameters
  ----------
  NRMSE_all : list
    Per-method lists of NRMSE curves versus ensemble size.
  NRMSE_all_base : list
    Baseline (noiseless) NRMSE curves for non-Lorenz cases.
  scenarios : list
    Scenario names used for colouring / legend.
  methods : list
    Method titles for subplot headings (length 4 when `cubLor` is False,
    length 6 when True).
  train : str
    Scenario used for training (ignored on diagonal plots).
  cubLor : bool, optional
    True for cubic-Lorenz plots (3 x 2 grid, log-y); False for
    box-model plots (2 x 2 grid, log-y).
  save_fig : bool, optional
    Save the figure as *Figures/<fig_name>.pdf* when True.
  fig_name : str, optional
    Output filename stem used if `save_fig` is True.

  Returns
  -------
  None
  """

  # Figure setup (varies by experiment)
  if cubLor:
    fig, ax = plt.subplots(3, 2, figsize=(14,8), sharex=True, sharey='row', layout='constrained')
    xaxis = np.arange(1, 5_001, 5_001//50)
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    x_shift, y_shift = -0.07, 1.18
  else:
    fig, ax = plt.subplots(2, 2, figsize=(14,8), sharex=True, sharey='row', layout='constrained')
    xaxis = np.arange(1,51)
    extra = [mlines.Line2D([], [], color='k', linestyle='-',  label='Noisy', lw=2, alpha=0.6),
           mlines.Line2D([], [], color='k', linestyle='--', label='Noiseless', lw=2, alpha=0.6)]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    x_shift, y_shift = -0.06, 1.11

  # Loop over subplots
  for i, ax in enumerate(ax.flat):
    NRMSE_ens = NRMSE_all[i]
    method = methods[i]

    if not cubLor:
      NRMSE_base = NRMSE_all_base[i]

    for j, scen in enumerate(scenarios):
      if train == scen:
        continue
      if cubLor:
        if method == 'Method II: FDT':
          ax.semilogy(xaxis, NRMSE_ens[scen], lw=2, c=cm.batlowS(j+3), label=scen)
          continue
        ax.semilogy(xaxis, NRMSE_ens[train][scen], lw=2, c=cm.batlowS(j+3), label=scen)

      else:
        ax.semilogy(xaxis, NRMSE_ens[train][scen], lw=2, c=cm.batlowS(j+3), label=scen)
        ax.axhline(y=np.mean(NRMSE_base[train][scen]),lw=2,ls='--', c=cm.batlowS(j+3))

    ax.set_title(f'{method}',fontsize=18)
    ax.grid()

    # Figure formatting
    if i == 0 and not cubLor:
      ax.legend(fontsize=16)
    elif i == 1 and not cubLor:
      ax.legend(extra, [h.get_label() for h in extra], fontsize=16)
    elif i == 0:
      ax.legend(fontsize=12)

    ax.text(x_shift, y_shift, panel_labels[i], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

  ax.tick_params(axis='both', which='major', labelsize=18)
  fig.supylabel(r'NRMSE [\%]',fontsize=24)
  fig.supxlabel('No. Ensemble Members',fontsize=24)
  fig.suptitle(f'NRMSE vs. Ensemble size by method\nTraining scenario: $\it{{{train}}}$',fontsize=24)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return

###############
## Figure 10 ##
###############

def plot_Lorenz_response(R_50k, R_5k, R_500, t, save_fig=False, fig_name=None):
  """
  Plot cubic-Lorenz response functions estimated with 500, 5,000, and
  50,000 ensemble members.

  Parameters
  ----------
  R_50k, R_5k, R_500 : ndarray
    Response functions for the three ensemble sizes.
  t : ndarray
    Time vector (years).
  save_fig : bool, optional
    Save PDF under *Figures/<fig_name>.pdf* when True.
  fig_name : str, optional
    Filename stem used if `save_fig` is True.

  Returns
  -------
  None
  """

  # Setup figure and plot
  fig, ax = plt.subplots(figsize=(12,4),layout='constrained')
  ax.plot(t, R_50k, lw=3, c=cm.batlowS(3), label='50,000 Members')
  ax.plot(t, R_5k, lw=3, c=cm.batlowS(4), ls = '-.', label='5,000 Members')
  ax.plot(t, R_500, lw=3, c=cm.batlowS(5), ls='--', label = '500 Members')
  legend = ax.legend()

  # Customize legend (right-aligned)
  max_shift = max([t.get_window_extent().width for t in legend.get_texts()])
  for t in legend.get_texts():
    t.set_ha('right')
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))

  # Figure formatting
  ax.set_ylabel(r'$\langle Z \rangle$ Response to $\Delta \rho$',fontsize=22)
  ax.set_xlabel('Year',fontsize=22)
  ax.tick_params(axis='both', which='major', labelsize=18)
  ax.set_title('Cubic Lorenz Response Function', fontsize=24)

  if save_fig:
    plt.savefig(f'Figures/{fig_name}.pdf',dpi=300)

  return