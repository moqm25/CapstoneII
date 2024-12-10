# Capstone II | CS493: WRF Model Optimization Codebase

Welcome to the repository dedicated to advancing the optimization of the Weather Research and Forecasting (WRF) model, specifically focusing on the WRF-Solar model enhancements. This project is part of the CS492 course and aims at refining parameter tuning methods to improve prediction accuracy and computational efficiency in climate modeling.

## Purpose of the Repository

This repository serves as a central hub for storing, documenting, and sharing innovative algorithms that optimize two critical parameters of the WRF-Solar model: `beta_con` and `vdis`. `beta_con` affects the microphysics related to cloud formation and condensation rates, while `vdis` influences the dispersion and scattering behaviors of aerosols and cloud particles in the model.

By minimizing the Mean Squared Error (MSE) between the model's predictions of solar irradiance (`SWDOWN`) and actual ground truth measurements, the algorithms aim to significantly enhance the model's reliability and utility for climate researchers and meteorologists.

## Repository Structure
**Key Directories and Files:**

- **`BO_algo.py`**: Implements Bayesian Optimization (BO) for parameter tuning.
- **`D3QN_algo.py`**: Implements a Double Deep Q-Network (D3QN) reinforcement learning approach for parameter optimization.
- **`SA_algo.py`**: Implements a Stochastic Approximation (SA) based method to iteratively improve parameters.

### Final Presentation

- **`Final Presentation.pdf`**: Contains a detailed presentation of the project, summarizing the background, objectives, research approach, key results, challenges, and areas for future work. This presentation provides a comprehensive recap of the project's outcomes and insights.

### Output Plots Directories

- **`BO_output_plots`**: Stores output plots generated by the Bayesian Optimization algorithm. Each subdirectory corresponds to a specific `(beta_con, vdis)` parameter combination and contains the resulting comparison plot.
  
- **`D3QN_output_plots`**: Stores output plots generated by the D3QN optimization runs. As with BO, each subdirectory represents a run with a particular `(beta_con, vdis)` and includes the performance plot.

- **`SA_output_plots`**: Contains plots from the Stochastic Approximation method, organized by parameter sets.

### Ground Truth Directory

- **`ground_truth`**: 
  - `ground_truth_data.npy`: Contains the ground truth data in NumPy binary format.
  - `ground_truth.jpg` & `ground_truth_subplot.jpg`: Visual representations of the ground truth data.
  - `sgpradflux10long_area_mean.c2.20090506_1200UTC.nc`: NetCDF file containing observational ground truth data for a specific date and time.

## Algorithms and Files Explained

### Bayesian Optimization (`BO_algo.py`)

- **Purpose**: Bayesian Optimization is used to find the best parameters `(beta_con, vdis)` for the WRF model by building a probabilistic model of the objective function (MSE). 
- **Key Steps**:
  1. **Update Namelist**: Adjusts the WRF `namelist.input` file with new `beta_con` and `vdis` values.
  2. **Run WRF Model**: Executes WRF with the updated parameters.
  3. **Process Output & Calculate MSE**: Extracts SWDOWN values from the WRF output and compares them with the ground truth data.
  4. **Optimize**: Uses Gaussian Processes (via `gp_minimize`) to iteratively propose better `(beta_con, vdis)` combinations, aiming to minimize MSE.
- **Outputs**: Saves plots and data in `BO_output_plots` for each evaluated parameter set.

### Double Deep Q-Network (`D3QN_algo.py`)

- **Purpose**: The D3QN algorithm treats parameter tuning as a reinforcement learning problem, where `(beta_con, vdis)` represent the state, and discretized adjustments to these parameters form the action space.
- **Key Steps**:
  1. **Q-Learning Setup**: Uses neural networks to approximate the Q-function, guiding policy updates.
  2. **Exploration vs. Exploitation**: Initially explores random actions (parameter changes), then exploits learned knowledge to pick better actions.
  3. **Reward Function**: Negative of MSE is used as a reward signal, encouraging actions leading to better model performance.
  4. **Experience Replay & Target Networks**: Enhances stability and performance of the RL approach.
- **Outputs**: Saves training progression plots under `D3QN_output_plots`. Each subdirectory indicates a distinct parameter run and the resulting MSE plot.

### Stochastic Approximation (`SA_algo.py`)

- **Purpose**: SA is a simpler iterative method where parameters `(beta_con, vdis)` are updated in a gradient-based manner. Unlike Bayesian Optimization, it does not maintain a global probabilistic model, and unlike D3QN, it doesn’t frame the problem as a reinforcement learning task.
- **Key Steps**:
  1. **Perturbation & Gradient Estimate**: Perturbs parameters slightly to estimate the gradient of MSE.
  2. **Parameter Updates**: Moves `(beta_con, vdis)` in the direction that reduces MSE.
  3. **Convergence Check**: Iterates until improvements in MSE plateau or a threshold is reached.
- **Outputs**: Similar to the other methods, generated plots are stored in `SA_output_plots` for each parameter configuration tested.

## Ground Truth Data Handling

- All methods rely on ground truth data stored in `ground_truth`.
- The MSE calculation involves comparing WRF output (SWDOWN) against the provided observational data (swdtot) to measure performance.

## How to Use This Repository

1. **Model Setup**: Ensure that the WRF model is correctly installed and configured on your system. [INSERT]
2. **Dependency Installation**: All dependencies should be taken care of in the initial install/Bash Script.
3. **Running the Algorithms**: Execute the algorithm scripts to perform optimizations. Each script is self-contained and provides options to adjust settings like the number of iterations, learning rates, and more.
4. **Evaluating Results**: Analyze the output plots and data to assess improvements in the model's performance and determine the best parameter configurations.
5. **Further Research**: Use the findings and methodologies as a basis for further academic research or practical applications in weather forecasting and climate studies.

## Collaborative and Educational Use

This repository is designed to be a resource for students, researchers, and professionals interested in climate modeling and numerical weather prediction. Users are encouraged to experiment with the algorithms, propose modifications, and contribute to the ongoing development of more refined optimization techniques.

We invite collaboration, discussion, and feedback to enhance the project's scope and impact.

## Running the Codes

- Before running any algorithm, ensure that:
  - WRF is correctly installed and compiled.
  - The `namelist.input` file path and other referenced directories are updated as per your local environment.
  - Required Python packages (`numpy`, `pandas`, `netCDF4`, `matplotlib`, `skopt`, `tensorflow` for D3QN, etc.) are installed.

- **Bayesian Optimization**: Run `python BO_algo.py`
- **D3QN**: Run `python D3QN_algo.py`
- **Stochastic Approximation**: Run `python SA_algo.py`

## Disclaimer

This repository is currently updated and maintained. Each method represents a unique approach to optimizing WRF model parameters. The code and strategies are subject to further refinements and improvements.
