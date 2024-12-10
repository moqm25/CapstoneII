import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For plotting
import netCDF4  # To handle WRF output files
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real  # Import Real to define parameter ranges
import shutil
from datetime import datetime

mse_values = []

# Function to update the namelist file
def update_namelist(file_path, beta_con, vdis):
    print(f"Updating namelist file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    
    for line in lines:
        if 'beta_con' in line:
            updated_line = f" beta_con = {beta_con:.2E},\n"
            updated_lines.append(updated_line)
        elif 'vdis' in line:
            updated_line = f" vdis = {vdis:.4f},\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)
    
    print(f"Updated namelist with beta_con={beta_con}, vdis={vdis}")

# Function to run the WRF model after updating the namelist on 6 cores
def run_WRF(wrf_exe_path, num_cores=6):
    wrf_dir = os.path.dirname(wrf_exe_path)
    print(f"Running WRF model in {wrf_dir} using {num_cores} cores...")

    wrf_run_result = subprocess.call(f"mpirun -np {num_cores} {wrf_exe_path}".split(), cwd=wrf_dir)
    
    if wrf_run_result == 0:
        print("WRF model run successfully.")
        return 1
    else:
        print("WRF model failed to run.")
        return 0

# Function to process the WRF output
def process_wrf_output(wrf_file):
    print(f"Processing WRF output file: {wrf_file}")
    return process_wrf_file(wrf_file)

# Copy WRF output files to the iteration-specific directory
def copy_wrf_output_files(wrf_dir, output_dir):
    files_to_copy = [
        os.path.join(wrf_dir, 'wrfout_d02_2016-06-19_06:00:00'),
        os.path.join(wrf_dir, 'wrfout_d01_2016-06-19_06:00:00')
    ]
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            shutil.copy(file_path, output_dir)
            print(f"Copied {file_path} to {output_dir}")
        else:
            print(f"File {file_path} not found")

# Function to process WRF data using the provided method
def process_wrf_file(nc_file):
    """Process SWDOWN data from a WRF NetCDF file."""
    with netCDF4.Dataset(nc_file, 'r') as nc:
        time_data = convert_time(nc, 'Times')  # Use 'Times' variable in WRF output
        swdown_data = np.mean(nc.variables['SWDOWN'][:, :, :], axis=(1, 2))  # Average over lat/lon
        data = np.column_stack((time_data, swdown_data))
    return data

# Function to convert NetCDF time variable to datetime objects
def convert_time(nc, time_var_name):
    time_var = nc.variables[time_var_name][:]  # 'Times' in WRF output
    time_data = [datetime.strptime(t.tobytes().decode('utf-8'), '%Y-%m-%d_%H:%M:%S') for t in time_var]
    return time_data

# Function to get ground truth data from the CSV
def process_ground_truth_csv(csv_file):
    df = pd.read_csv(csv_file)
    ground_truth_time = pd.to_datetime(df['period_end'])
    ground_truth_data = df['swdtot'].values
    return ground_truth_time, ground_truth_data

# Function to calculate the Mean Squared Error (MSE)
def calculate_mse(wrf_data, ground_truth_data):
    wrf_swdown_data = wrf_data[:, 1].astype(float)
    
    if len(wrf_swdown_data) != len(ground_truth_data):
        print(f"Resampling WRF data from {len(wrf_swdown_data)} to match ground truth {len(ground_truth_data)}")
        wrf_swdown_data = np.interp(np.linspace(0, len(wrf_swdown_data) - 1, len(ground_truth_data)), 
                                    np.arange(len(wrf_swdown_data)), wrf_swdown_data)

    mse_result = mean_squared_error(ground_truth_data, wrf_swdown_data)
    print(f"MSE between WRF SWDOWN and Ground Truth: {mse_result}")
    return mse_result

# Function to plot comparison between WRF output and ground truth data
def plot_comparison(wrf_data, ground_truth_time, ground_truth_data, output_dir):
    wrf_time_data = wrf_data[:, 0]
    wrf_swdown_data = wrf_data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(wrf_time_data, wrf_swdown_data, label='WRF SWDOWN', linestyle='--')
    plt.plot(ground_truth_time, ground_truth_data, label='Ground Truth swdtot', linestyle='--')
    
    plt.xlabel('Time')
    plt.ylabel('Mean SWDOWN')
    plt.title('Comparison of SWDOWN from WRF and Ground Truth Data')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, 'wrf_vs_ground_truth.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved as: {plot_filename}")

# Objective function for optimization
def objective(file_path, params, GTDIR, wrf_file, csv_file, wrf_exe_path):
    beta_con, vdis = params
    # New directory for each iteration based on beta_con and vdis
    output_dir = f"/home/capstonei/Sohaib_Project2/WRFComputerSteering/iterations/beta_{beta_con}_vdis_{vdis}"

    # Check if the output directory for this combination already exists (skip if it does)
    if os.path.exists(output_dir):
        print(f"Skipping beta_con={beta_con}, vdis={vdis} as it has already been processed.")
        return 1e6  # Return a large error to skip this combination
    
    # Update namelist file
    update_namelist(file_path, beta_con, vdis)

    # Run WRF model on 6 cores
    wrf_run_success = run_WRF(wrf_exe_path, num_cores=6)

    if wrf_run_success:
        # Process WRF output
        wrf_data = process_wrf_output(wrf_file)

        # Get ground truth data
        gt_time, gt_data = process_ground_truth_csv(csv_file)

        # Calculate Mean Squared Error
        mse_result = calculate_mse(wrf_data, gt_data)
        mse_values.append(mse_result)

        # Copy output files to the iteration-specific directory
        copy_wrf_output_files(os.path.dirname(wrf_exe_path), output_dir)

        # Plot comparison between WRF output and ground truth
        plot_comparison(wrf_data, gt_time, gt_data, output_dir)

        print(f"Parameters: beta_con={beta_con}, vdis={vdis}, MSE={mse_result}")
        return mse_result
    else:
        print("WRF run failed, returning high MSE.")
        return 1e6  # Return a large error in case of failure

# Main optimization function without checkpointing
def optimize(namelist_file_path, wrf_exe_path, GTDIR, wrf_file, csv_file):
    space = [
        Real(1.0e20, 1.67e24, name='beta_con'),
        Real(0.01, 1.4, name='vdis')
    ]

    print("Starting optimization from scratch...")
    result = gp_minimize(
        func=lambda params: objective(namelist_file_path, params, GTDIR, wrf_file, csv_file, wrf_exe_path),
        dimensions=space,
        n_calls=50,  # Number of iterations
        random_state=123
    )

    return result

if __name__ == '__main__':
    namelist_file_path = "/home/capstonei/Sohaib_Project2/WRFComputerSteering/cumulous_case/WRF_Input/namelist.input"
    GTDIR = '/home/capstonei/Sohaib_Project2/WRFComputerSteering/ground_truth/'
    wrf_exe_path = "/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/wrf.exe"
    OUTPUT_DIR = "/home/capstonei/Sohaib_Project2/WRFComputerSteering/output_plots/"
    csv_file = '/home/capstonei/Sohaib_Project2/WRFComputerSteering/ground_truth/cumulousGroundTruth.csv'
    wrf_file = '/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/wrfout_d02_2016-06-19_06:00:00'

    # Run the optimization without checkpointing
    result = optimize(
        namelist_file_path,
        wrf_exe_path,
        GTDIR,
        wrf_file,
        csv_file
    )

    print(f"Final optimization result: {result}")