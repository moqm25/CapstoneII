import os
import random
import subprocess
import shutil
import netCDF4
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# WRF Parameters and Stochastic Approximation settings
class StochasticApproximation:
    def __init__(self):
        self.beta_con_min, self.beta_con_max = 1.0e20, 1.67e24
        self.vdis_min, self.vdis_max = 0.01, 1.4
        self.delta_beta = 5e21
        self.delta_vdis = 0.01
        self.alpha = 0.05  # Learning rate for stochastic approximation
        self.base_dir = "/home/capstonei/WRF-Computer-Steering/iterations_2"

    def update_params(self, beta_con, vdis, gradient_beta, gradient_vdis):
        """Update parameters using a stochastic gradient descent approach."""
        beta_con = beta_con - self.alpha * gradient_beta
        vdis = vdis - self.alpha * gradient_vdis

        # Clamp values to their bounds
        beta_con = min(max(beta_con, self.beta_con_min), self.beta_con_max)
        vdis = min(max(vdis, self.vdis_min), self.vdis_max)
        return beta_con, vdis

    def reward_function(self, current_mse, prev_mse):
        if prev_mse is None:
            return 0
        return (prev_mse - current_mse) / (prev_mse + 1e-8)

    def calculate_gradient(self, current_mse, prev_mse, delta_param):
        """Calculate gradient estimate for stochastic approximation."""
        if prev_mse is None:
            return 0
        return (current_mse - prev_mse) / delta_param

    def train(self, episodes, max_steps_per_episode):
        for episode in range(episodes):
            # Initialize parameters randomly
            beta_con = random.uniform(self.beta_con_min, self.beta_con_max)
            vdis = random.uniform(self.vdis_min, self.vdis_max)
            prev_mse = None
            step_count = 0

            while step_count < max_steps_per_episode:
                # Save current parameters
                beta_con_prev = beta_con
                vdis_prev = vdis

                # Perturb parameters for gradient estimation
                beta_con_perturbed = beta_con + random.uniform(-self.delta_beta, self.delta_beta)
                vdis_perturbed = vdis + random.uniform(-self.delta_vdis, self.delta_vdis)

                # Evaluate model performance for current parameters
                current_mse = self.run_simulation(beta_con_perturbed, vdis_perturbed)

                # Compute gradients using finite differences
                gradient_beta = self.calculate_gradient(current_mse, prev_mse, self.delta_beta)
                gradient_vdis = self.calculate_gradient(current_mse, prev_mse, self.delta_vdis)

                # Update parameters using stochastic approximation
                beta_con, vdis = self.update_params(beta_con, vdis, gradient_beta, gradient_vdis)

                # Print progress
                print(f"Episode {episode + 1}, Step {step_count + 1}: MSE = {current_mse:.2f}, "
                      f"beta_con = {beta_con:.2e}, vdis = {vdis:.4f}")

                # Check for convergence
                if current_mse < 200:
                    print("Converged to optimal solution.")
                    break

                prev_mse = current_mse
                step_count += 1

    def run_simulation(self, beta_con, vdis):
        """Run the WRF model and compute MSE."""
        iteration_dir = f"{self.base_dir}/beta_{beta_con:.2e}_vdis_{vdis:.4f}"
        os.makedirs(iteration_dir, exist_ok=True)

        self.update_namelist(
            "/home/capstonei/WRF-Computer-Steering/cumulous_case/WRF_Input/namelist.input",
            beta_con, vdis
        )

        if self.run_wrf_model():
            wrf_data = self.process_wrf_output('/home/capstonei/WRF-Computer-Steering/Build_WRF/WRF/run/wrfout_d02_2016-06-19_06:00:00')
            gt_time, gt_data = self.process_ground_truth_csv('/home/capstonei/WRF-Computer-Steering/ground_truth/cumulousGroundTruth.csv')
            mse = self.calculate_mse(wrf_data, gt_data)
            self.copy_wrf_output_files(iteration_dir)
            self.plot_comparison(wrf_data, gt_time, gt_data, iteration_dir)
            return mse
        else:
            print("Simulation failed.")
            return float('inf')

    def update_namelist(self, file_path, beta_con, vdis):
        """Update the WRF namelist.input file."""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        updated_lines = []
        for line in lines:
            if 'beta_con' in line:
                updated_lines.append(f" beta_con = {beta_con:.2E},\n")
            elif 'vdis' in line:
                updated_lines.append(f" vdis = {vdis:.4f},\n")
            else:
                updated_lines.append(line)
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

    def process_wrf_output(self, nc_file):
        with netCDF4.Dataset(nc_file, 'r') as nc:
            time_data = [datetime.strptime(t.tobytes().decode('utf-8'), '%Y-%m-%d_%H:%M:%S') for t in nc.variables['Times'][:]]
            swdown_data = np.mean(nc.variables['SWDOWN'][:, :, :], axis=(1, 2))
            return np.column_stack((time_data, swdown_data))

    def calculate_mse(self, wrf_data, ground_truth_data):
        wrf_swdown_data = wrf_data[:, 1].astype(float)
        ground_truth_data = ground_truth_data.astype(float)

        if len(wrf_swdown_data) != len(ground_truth_data):
            wrf_swdown_data = np.interp(
                np.linspace(0, len(wrf_swdown_data) - 1, len(ground_truth_data)),
                np.arange(len(wrf_swdown_data)),
                wrf_swdown_data
            )

        mse_result = np.mean((wrf_swdown_data - ground_truth_data) ** 2)
        return mse_result

    def copy_wrf_output_files(self, output_dir):
        files_to_copy = [
            '/home/capstonei/WRF-Computer-Steering/Build_WRF/WRF/run/wrfout_d02_2016-06-19_06:00:00',
            '/home/capstonei/WRF-Computer-Steering/Build_WRF/WRF/run/wrfout_d01_2016-06-19_06:00:00'
        ]
        os.makedirs(output_dir, exist_ok=True)
        for file_path in files_to_copy:
            if os.path.exists(file_path):
                shutil.copy(file_path, output_dir)
                print(f"Copied {file_path} to {output_dir}")
            else:
                print(f"File {file_path} not found")

    def plot_comparison(self, wrf_data, ground_truth_time, ground_truth_data, output_dir):
        wrf_time_data = wrf_data[:, 0]
        wrf_swdown_data = wrf_data[:, 1]
        plt.figure(figsize=(10, 6))
        plt.plot(wrf_time_data, wrf_swdown_data, label='WRF SWDOWN', linestyle='--')
        plt.plot(ground_truth_time, ground_truth_data, label='Ground Truth swdtot', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean SWDOWN')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'wrf_vs_ground_truth.png'))
        plt.close()

    def process_ground_truth_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        ground_truth_time = pd.to_datetime(df['period_end'])
        ground_truth_data = df['swdtot'].values
        return ground_truth_time, ground_truth_data

    def run_wrf_model(self):
        wrf_run_result = subprocess.call(
            "mpirun -np 6 ./wrf.exe".split(),
            cwd="/home/capstonei/WRF-Computer-Steering/Build_WRF/WRF/run/"
        )
        return wrf_run_result == 0


# Train the model with Stochastic Approximation
sa = StochasticApproximation()
sa.train(episodes=5, max_steps_per_episode=7)
