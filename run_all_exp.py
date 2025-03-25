import BudykoSellers
import emulator_utils
import numpy as np
import csv

def run_all(basis, degree, noisy=False, verbose=False):
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

  experiments = ['3box_uncoup','3box_coup','2box_coup','2box_coup_no_ocean']
  scenarios = ['2xCO2','High Emissions','Overshoot']
  methods = ['DMD','EDMD','deconvolve','direct']

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
      temp_outputs = BudykoSellers.Run_Budyko_Sellers(exp_flag=i,
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
      # Initial condition
      w0 = np.zeros(n_boxes)

      operator[method][exp], T_pred[method][exp], error_metrics[method][exp] = emulator_utils.emulate_scenarios(method, scenarios=scenarios, outputs=T_out[exp], forcings=forcings[exp], w0=w0, t=t, dt=dt, n_steps=n_steps, n_boxes=n_boxes, w_dict=w_dict, F_dict=F_dict, verbose=verbose)

  # Emulate with Fitting Modes
  ## To do

  data = [['Method','Experiment','Scenario - Train','Scenario - Test','Region','RMSE','MAE','Bias','MRE']]

  for method in methods:
    for i, exp in enumerate(experiments):
      n_boxes = get_num_boxes(i)
      for scen1 in scenarios:
        if method == 'direct':
          for reg in range(n_boxes):
            RMSE = error_metrics[method][exp][scen1][0][reg]
            MAE = error_metrics[method][exp][scen1][1][reg]
            Bias = error_metrics[method][exp][scen1][2][reg]
            MRE = error_metrics[method][exp][scen1][3][reg]
            data.append([method,exp,scen1,'n/a',reg,RMSE,MAE,Bias,MRE])

        else:
          for scen2 in scenarios:
            for reg in range(n_boxes):
              RMSE = error_metrics[method][exp][scen1][scen2][0][reg]
              MAE = error_metrics[method][exp][scen1][scen2][1][reg]
              Bias = error_metrics[method][exp][scen1][scen2][2][reg]
              MRE = error_metrics[method][exp][scen1][scen2][3][reg]
              data.append([method,exp,scen1,scen2,reg,RMSE,MAE,Bias,MRE])

  # Output data to CSV
  file_path = 'deterministic_results.csv'

  # Open the file in write mode ('w')
  with open(file_path, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write multiple rows using writerows()
    writer.writerows(data)

  return

def get_num_boxes(i):
  # 3 box vs. 2 box
  if i <= 1:
    n_boxes = 3
  else:
    n_boxes = 2
  return n_boxes

#basis, degree = 'hermite', 1
#run_all(basis, degree)