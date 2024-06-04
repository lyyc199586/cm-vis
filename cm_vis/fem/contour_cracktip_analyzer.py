import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class VelocityAnalyzer:
    def __init__(self, directory):
        self.directory = directory
        self.filter_condition = None
        self.window_length = None
        self.polyorder = None
        self.tip_list = self._load_tip_coords()
        
    def set_savgol_params(self, windows_length, polyorder):
        self.window_length = windows_length
        self.polyorder = polyorder
        
    def _extract_step(self, filename):
        # extract time steps of contour csv file
        match = re.search(r'contour_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Filename {filename} does not match the expected pattern 'contour_<number>'.")
    
    def _load_tip_coords(self):
        files = glob.glob(f"{self.directory}/*")
        files.sort(key=self._extract_step)
        tip_list = []
        for file in files:
            tip = self._tip_coords(file)
            tip_list.append(tip)
        return np.array(tip_list)
    
    def _tip_coords(self, file):
        df = pd.read_csv(file)
        if self.filter_condition:
            filtered_df = df.query(self.filter_condition).copy()
        else:
            filtered_df = df.copy()
        filtered_df['x^2 + y^2'] = filtered_df['Points:0']**2 + filtered_df['Points:1']**2
        max_row = filtered_df.loc[filtered_df['x^2 + y^2'].idxmax()]
        return [max_row['Time'], max_row['Points:0'], max_row['Points:1']]
      
    def calc_crack_length(self):
        lengths = [0.0]
        for i in range(1, len(self.tip_list)):
            dx = np.sqrt((self.tip_list[i, 1] - self.tip_list[i - 1, 1])**2 + (self.tip_list[i, 2] - self.tip_list[i - 1, 2])**2)
            lengths.append(lengths[i - 1] + dx)
        return np.array(lengths)

    def calc_vel_direct(self):
        velocities = [0.0]
        for i in range(1, len(self.tip_list)):
            dt = self.tip_list[i, 0] - self.tip_list[i - 1, 0]
            dx = np.sqrt((self.tip_list[i, 1] - self.tip_list[i - 1, 1])**2 + (self.tip_list[i, 2] - self.tip_list[i - 1, 2])**2)
            v = dx / dt
            velocities.append(v)
        return np.array(velocities)
    
    def calc_vel_savgol(self):
        if self.window_length is None or self.polyorder is None:
            raise ValueError("Savitzky-Golay parameters not set. Use set_savgol_params to set them.")
        t = self.tip_list[:, 0]
        x = self.tip_list[:, 1]
        y = self.tip_list[:, 2]
        smoothed_x = savgol_filter(x, self.window_length, self.polyorder)
        smoothed_y = savgol_filter(y, self.window_length, self.polyorder)
        velocities = [0.0]
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            dx = np.sqrt((smoothed_x[i] - smoothed_x[i - 1])**2 + (smoothed_y[i] - smoothed_y[i - 1])**2)
            v = dx / dt
            velocities.append(v)
        return np.array(velocities)
    
    def analyze(self):
        lengths = self.calc_crack_length()
        velocities_direct = self.calc_vel_direct()
        velocities_filtered = self.calc_vel_savgol()
        combined_output = np.hstack((self.tip_list, lengths.reshape(-1, 1), velocities_direct.reshape(-1, 1), velocities_filtered.reshape(-1, 1)))
        return combined_output
    
    def plot_velocities(self, data):
        t = data[:, 0]
        v_direct = data[:, 3]
        v_filtered = data[:, 4]
        fig, ax = plt.subplots()
        ax.plot(t, v_direct, label='Direct Velocity')
        ax.plot(t, v_filtered, label='Filtered Velocity', linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Velocity')
        ax.legend()
        plt.show()
        
    def save_to_csv(self, data, path=None):
        if path is None:
            path = os.path.join(self.directory, 'velocity_output.csv')
        df_combined = pd.DataFrame(data, columns=['time', 'x', 'y', 'crack_length', 'vel_direct', 'vel_filtered'])
        df_combined.to_csv(path, index=False)