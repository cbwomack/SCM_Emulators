# Experiments for Stress Testing Climate Emulators

In this repository, you'll find a series of experiments using simple climate models to stress test a several popular (and a few lesser-known) climate emulation techniques. This is the companion code for [insert name when we finalize], and the preprint corresponding to this work can be found here (add link after submission).

# Usage
## Jupyter Notebooks and Code
All code for training and running the emulators is written in python, specifically within the Jupyter notebooks included in this repo. This code does not require many non-standard python packages, with the exception of [JAX](https://docs.jax.dev/en/latest/quickstart.html) for optimization, along with [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) for plotting. 

Notebooks are organized as:
1. Experiment 1: Coupled three box model - tests the impact of memory effects.
2. Experiment 2: Restricted two box model - tests the impact of hidden variables.
3. Experiment 3: Noisy three box model - tests the impact of noise.
4. Experiment 4: Cubic Lorenz model - tests the impact of nonlinearities and noise.
5. Create plots: Creates all plots for the manuscript associated with this work.

Also included are 'utils_*.py' files that are required for the notebooks to function properly. These contain helper functions for e.g. running the simple climate models, cubic Lorenz system, and plotting scripts.

### Emulators
We test a suite of six emulators across the experiments and scenarios considered in this work:
| Technique                                  | Short Description                                                    | Key Assumptions                                                                      |
|--------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Method I: Pattern Scaling                  | Time-invariant pattern based on global mean temperature              | Climate is always near equilibrium; response is instantaneous; fixed spatial pattern |
| Method II: Fluctuation Dissipation Theorem | Response functions derived through perturbation ensemble experiments | Perturbations are small; data come from linear response regime                       |
| Method III: Deconvolution                  | Response functions solved for from any general experiment            | Quasi-equilibrium initial condition; influence of noise is small                     |
| Method IV: Modal Fitting                   | Response functions fit from any general experiment                   | Response is a decaying exponential; few significant modes                            |
| Method V: Dynamic Mode Decomposition (DMD) | Approximating system dynamics with a linear operator                 | Dynamics are approx. linear; data capture relevant dynamics                          |
| Method VI: Extended DMD                    | Approximating system dynamics with nonlinear basis functions         | Basis functions span Koopman operator; dynamics are approx. linear in new basis      |

### Scenarios
Each experiment has a consistent set of four scenarios, with parameters that differ between the simple box model and the cubic Lorenz system:
| Scenario         | Short Description                                                                                                                                                |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _Abrupt_         | An abrupt doubling of CO concentration; corresponds roughly to the _Abrupt2xCO2_ CMIP experiment.                                                                |
| _High Emissions_ | An exponential increase of CO2 concentration in time; corresponds roughly to _SSP585_.                                                                           |
| _Plateau_        | An increase in CO2 concentration in time that follows a hyperbolic tangent, increasing exponentially and then tapering off; corresponds roughly to _SSP245_.     |
| _Overshoot_      | An increase in CO2 concentration in time that follows a Gaussian profile, increasing and decreasingly rapidly; inspired by _SSP119_, but decreases more quickly. |

See manuscript for more details on experiments, emulators, and scenarios.

## File Structure
All code necessary for running the actual experiments are present in the main directory. The subfolders 'Figures/' and 'Results/' are used to store the graphical outputs and data from running the emulators, respectively.

The folder 'utils_plot_fig01' contains the scripts necessary for producing Figure 1 of the manuscript. These scripts require additional data not included in this repository, due to the size of the files involved. These data are taken from the MPI Grand Ensemble ([Maher et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001639)), and can be downloaded from the [Earth System Grid Federation](https://aims2.llnl.gov/search/cmip6/).
