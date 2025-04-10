import BudykoSellers
import emulator_utils
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def run_all(basis, degree, methods, experiments, scenarios, n_modes=1, noisy=False, verbose=False, save=False):
  """
  Runs all experiments and outputs the emulation
  results as a CSV.
  """
  if noisy:
    raise ValueError('Error, noisy experiments not implemented.')

  t_end = 251
  t = np.arange(0,t_end)
  dt = 1
  n_steps = len(t)
  w_dict = emulator_utils.Vector_Dict(method=basis, degree=degree)
  F_dict = emulator_utils.Vector_Dict(method=basis, degree=degree)

  forcings, T_out = {}, {}
  operator, T_pred, error_metrics = {}, {}, {}

  # Simulate every experiment
  print('Running Experiments...')
  for i, exp in enumerate(experiments):
    forcings[exp], T_out[exp] = {}, {}

    n_boxes = get_num_boxes(i)

    # Horizontally coupled vs. not
    if i == 1:
      diff_flag = 1
    else:
      diff_flag = 0

    # Vertically coupled vs not
    if i > 1:
      vert_diff_flag = 1
    else:
      vert_diff_flag = 0

    for j, scen in enumerate(scenarios):
      temp_outputs = BudykoSellers.Run_Budyko_Sellers(scen_flag=j,
                                                      n_boxes=n_boxes,
                                                      diff_flag=diff_flag,
                                                      vert_diff_flag=vert_diff_flag)
      forcings[exp][scen] = np.tile(temp_outputs['forcing_ts'], (n_boxes, 1))
      T_out[exp][scen] = np.squeeze(temp_outputs['T_ts'])[0:n_boxes,:]

      # Zero out oceanic forcing
      if vert_diff_flag == 1:
        forcings[exp][scen][1] = np.zeros(len(forcings[exp][scen][1]))

  # Emulate scenarios
  for method in methods:
    print(f'Emulating with {method}...')
    operator[method], T_pred[method], error_metrics[method] = {}, {}, {}
    for i, exp in enumerate(experiments):
      n_boxes = get_num_boxes(i)
      n_modes = n_boxes
      # Initial condition
      w0 = np.zeros(n_boxes)

      # Horizontally coupled vs. not
      if i == 1:
        diff_flag = 1
      else:
        diff_flag = 0

      # Vertically coupled vs not
      if i > 1:
        vert_diff_flag = 1
      else:
        vert_diff_flag = 0

      operator[method][exp], T_pred[method][exp], error_metrics[method][exp] = emulator_utils.emulate_scenarios(method, scenarios=scenarios, outputs=T_out[exp], forcings=forcings[exp], w0=w0, t=t, dt=dt, n_steps=n_steps, n_boxes=n_boxes, w_dict=w_dict, F_dict=F_dict, verbose=verbose, B=np.ones(n_boxes), n_modes=n_modes, diff_flag=diff_flag, vert_diff_flag=vert_diff_flag)

  data = [['Method','Experiment','Scenario - Train','Scenario - Test','Region','NRMSE']]

  for method in methods:
    for i, exp in enumerate(experiments):
      n_boxes = get_num_boxes(i)
      for scen1 in scenarios:
        if method == 'direct':
          for reg in range(n_boxes):
            NRMSE = error_metrics[method][exp][scen1][reg]
            data.append([method,exp,scen1,'n/a',reg,NRMSE])

        else:
          for scen2 in scenarios:
            for reg in range(n_boxes):
              NRMSE = error_metrics[method][exp][scen1][scen2][reg]
              data.append([method,exp,scen1,scen2,reg,NRMSE])

  # Output data to CSV
  file_path = 'deterministic_results.csv'
  if save:
    # Open the file in write mode ('w')
    with open(file_path, 'w', newline='') as file:
      # Create a CSV writer object
      writer = csv.writer(file)

      # Write multiple rows using writerows()
      writer.writerows(data)

  return error_metrics

def get_num_boxes(i):
  # 3 box vs. 2 box
  if i <= 1:
    n_boxes = 3
  else:
    n_boxes = 2
  return n_boxes

def plot_error_heatmaps(error_metrics, exp, regions,
                        methods, train_scenarios, test_scenarios,
                        error_labels=None):
  """
  Generate a grid of heatmaps for a given experiment and region.

  Parameters
  ----------
  error_metrics : dict
    Nested dictionary with structure:
    error_metrics[method][exp][scen1][scen2][i][reg].
  exp : str
    The experiment name (e.g. "2box" or "3box").
  region : str
    The region name/key to extract from error_metrics[...][reg].
  methods : list of str
    List of method keys to display (each becomes a row in the grid).
  train_scenarios : list of str
    List of training-scenario keys (will form the y-axis).
  test_scenarios : list of str
    List of testing-scenario keys (will form the x-axis).
  error_labels : list of str, optional
    Labels for each error metric i=0..3. If omitted, uses generic labels.
  """

  # Number of error metrics is assumed to be 4 (i = 0..3).
  n_regions = len(regions)

  # Prepare the figure: rows = number of methods, cols = number of error metrics
  fig, axes = plt.subplots(
    nrows=n_regions, ncols=len(methods),
    figsize=(5 * len(methods), 6 * n_regions),
    constrained_layout=True, sharex='col', sharey='row')

  labels=['0','1','2']

  for region_idx in range(n_regions):
    cmap = 'Reds'

    # Get position of the *last* axis in this row (rightmost column)
    pos = axes[region_idx, -1].get_position()
    print(pos)
    if region_idx == 0:
      yshift = 0.11
      xshift = 0.11
    else:
      yshift = 0
      xshift = 0.01
    cbar_ax = fig.add_axes([
        pos.x1 + xshift,  # shift right
        pos.y0 + yshift,         # align vertically with the row
        0.015,          # width of colorbar
        pos.height      # height of the row
    ])

    # Get min and max values for colorbar
    min_val, max_val = np.inf, -np.inf
    for method in methods:
      if method == 'direct':
        for scen1 in train_scenarios:
          value = error_metrics[method][exp][scen1][region_idx]
          if value < min_val:
            min_val = value
          if value > max_val:
            max_val = value
      else:
        for scen1 in train_scenarios:
          for scen2 in test_scenarios:
            value = error_metrics[method][exp][scen1][scen2][region_idx]
            if value < min_val:
              min_val = value
            if value > max_val:
              max_val = value

    if max_val > 10:
      max_val = 10

    for col_idx, method in enumerate(methods):
      ax = axes[region_idx, col_idx]

      # Build the 2D matrix of errors: shape (len(train_scenarios), len(test_scenarios))
      data_matrix = []
      for scen1 in train_scenarios:
        row_values = []
        for scen2 in test_scenarios:
          if method == 'direct' and scen1 == scen2:
            value = error_metrics[method][exp][scen1][region_idx]
            row_values.append(value)
          elif method == 'direct':
            row_values.append(0)
          else:
            value = error_metrics[method][exp][scen1][scen2][region_idx]
            row_values.append(value)
        data_matrix.append(row_values)

        im = sns.heatmap(
          np.array(data_matrix).T, ax=ax, vmin=min_val, vmax=max_val, annot=True,
          cmap=cmap, linewidth=0.5, yticklabels=labels,
          xticklabels=labels, cbar_ax=cbar_ax)

        # Titles
        if col_idx == 0:
          ax.set_ylabel(regions[region_idx])
        if region_idx == 0:
          ax.set_title(method)

  return


#basis, degree = 'hermite', 1
#run_all(basis, degree)