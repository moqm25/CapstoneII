import os
import random
import subprocess
import shutil
import netCDF4
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from collections import deque
import json

# Set up GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

actions = [0, 1, 2, 3, 4, 5]  # Updated actions


class D3QN:
    def __init__(self, state_size, action_size, experience_file='experiences.json'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1  # Reduced minimum exploration
        self.epsilon_decay = 0.98  # Faster decay
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # WRF and parameter settings
        self.beta_con_min, self.beta_con_max = 1.0e20, 1.67e24
        self.vdis_min, self.vdis_max = 0.01, 1.4
        self.delta_beta = 5e21
        self.delta_vdis = 0.01
        self.base_dir = "/home/capstonei/Sohaib_Project2/WRFComputerSteering/iterations_2"
        self.visited_combinations = set()  # Store visited (beta_con, vdis) pairs
        self.experience_file = experience_file

        # Ensure the experience file is initialized
        self._initialize_experience_file()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _initialize_experience_file(self):
        if not os.path.exists(self.experience_file):
            with open(self.experience_file, 'w') as f:
                json.dump([], f)

    def remember(self, state, action, reward, next_state, done):
        # Convert NumPy types to native Python types
        experience = {
            "state": [float(x) for x in state[0]],  # Convert state to a list of floats
            "action": int(action),                 # Ensure action is an int
            "reward": float(reward),               # Ensure reward is a float
            "next_state": [float(x) for x in next_state[0]],  # Convert next_state to a list of floats
            "done": bool(done)                     # Ensure done is a bool
        }
        self.memory.append(experience)

        # Load existing data from the experience file
        try:
            with open(self.experience_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

    # Append the new experience to the existing data
        existing_data.append(experience)

    # Write back the combined data to the file
        with open(self.experience_file, 'w') as f:
            json.dump(existing_data, f)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Exploit

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for experience in minibatch:
            state = np.array(experience['state']).reshape(1, -1)
            action = experience['action']
            reward = experience['reward']
            next_state = np.array(experience['next_state']).reshape(1, -1)
            done = experience['done']

            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def reward_function(self, current_mse, prev_mse):
        if prev_mse is None:
            return 0
        return (prev_mse - current_mse) / (prev_mse + 1e-8)

    def take_action(self, action, beta_con, vdis):
        if action == 0:
            beta_con = min(beta_con + self.delta_beta, self.beta_con_max)
        elif action == 1:
            beta_con = max(beta_con - self.delta_beta, self.beta_con_min)
        elif action == 2:
            vdis = min(vdis + self.delta_vdis, self.vdis_max)
        elif action == 3:
            vdis = max(vdis - self.delta_vdis, self.vdis_min)
        elif action == 4:
            beta_con = min(beta_con + self.delta_beta, self.beta_con_max)
            vdis = min(vdis + self.delta_vdis, self.vdis_max)
        elif action == 5:
            beta_con = max(beta_con - self.delta_beta, self.beta_con_min)
            vdis = max(vdis - self.delta_vdis, self.vdis_min)
        return beta_con, vdis

    def train(self, episodes, batch_size, max_steps_per_episode):
        for episode in range(episodes):
            beta_con, vdis = self.initialize_params()
            state = np.array([beta_con, vdis]).reshape(1, -1)
            done = False
            step_count = 0
            prev_mse = None

            with open(self.experience_file, 'r') as f:
                try:
                    experiences = json.load(f)
                except json.JSONDecodeError:
                    experiences = []

            while not done and step_count < max_steps_per_episode:
                action = self.act(state)
                new_beta_con, new_vdis = self.take_action(action, beta_con, vdis)

                if any(exp["state"] == [new_beta_con, new_vdis] for exp in experiences):
                    mse = next(exp["reward"] for exp in experiences if exp["state"] == [new_beta_con, new_vdis])
                    print(f"Skipping simulation for beta_con: {new_beta_con}, vdis: {new_vdis}. Retrieved MSE: {mse}")
                    next_state = np.array([new_beta_con, new_vdis]).reshape(1, -1)
                    reward = mse
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    beta_con, vdis = new_beta_con, new_vdis
                    step_count += 1
                    continue

                iteration_dir = f"{self.base_dir}/beta_{new_beta_con:.2e}_vdis_{new_vdis:.4f}"
                os.makedirs(iteration_dir, exist_ok=True)

                self.update_namelist(
                    "/home/capstonei/Sohaib_Project2/WRFComputerSteering/cumulous_case/WRF_Input/namelist.input",
                    new_beta_con, new_vdis
                )

                if self.run_wrf_model():
                    wrf_data = self.process_wrf_output('/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/wrfout_d02_2016-06-19_06:00:00')
                    gt_time, gt_data = self.process_ground_truth_csv('/home/capstonei/Sohaib_Project2/WRFComputerSteering/ground_truth/cumulousGroundTruth.csv')
                    mse = self.calculate_mse(wrf_data, gt_data)

                    reward = self.reward_function(mse, prev_mse)
                    prev_mse = mse

                    next_state = np.array([new_beta_con, new_vdis]).reshape(1, -1)
                    done = mse < 200

                    self.remember(state, action, reward, next_state, done)
                    experiences.append({
                        "state": [float(x) for x in [new_beta_con, new_vdis]],
                        "action": int(action),
                        "reward": float(mse),
                        "next_state": [float(x) for x in next_state[0]],
                        "done": bool(done)
                    })

                    with open(self.experience_file, 'w') as f:
                        json.dump(experiences, f)

                    self.copy_wrf_output_files(iteration_dir)
                    self.plot_comparison(wrf_data, gt_time, gt_data, iteration_dir)

                    state = next_state
                    beta_con, vdis = new_beta_con, new_vdis

                    print(f"Episode {episode + 1}, Step {step_count + 1} with MSE: {mse}, beta_con: {new_beta_con}, vdis: {new_vdis}")

                step_count += 1

            self.update_target_model()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"Finished episode {episode + 1} with {step_count} steps")

        self.save('d3qn_model.h5')
        print("Training complete. Model saved.")

    def update_namelist(self, file_path, beta_con, vdis):
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

        mse_result = mean_squared_error(ground_truth_data, wrf_swdown_data)
        return mse_result

    def copy_wrf_output_files(self, output_dir):
        files_to_copy = [
            '/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/wrfout_d02_2016-06-19_06:00:00',
            '/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/wrfout_d01_2016-06-19_06:00:00'
        ]
        os.makedirs(output_dir, exist_ok=True)
        for file_path in files_to_copy:
            if os.path.exists(file_path):
                shutil.copy(file_path, output_dir)
                print(f"Copied {file_path} to {output_dir}")
            else:
                print(f"File {file_path} not found")

    def initialize_params(self):
        beta_con = random.uniform(self.beta_con_min, self.beta_con_max)
        vdis = random.uniform(self.vdis_min, self.vdis_max)
        return beta_con, vdis

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
            cwd="/home/capstonei/Sohaib_Project2/WRFComputerSteering/Build_WRF/WRF/run/"
        )
        return wrf_run_result == 0


state_size = 2
action_size = len(actions)
agent = D3QN(state_size, action_size)

# Train the model
agent.train(episodes=5, batch_size=16, max_steps_per_episode=7)
