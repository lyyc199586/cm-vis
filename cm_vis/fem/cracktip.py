import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class CrackTipAnalyzer:
    def __init__(self, filepath: str):
        """
        Initialize the CrackTipAnalyzer with a file containing raw crack tip trajectory.
        :param filepath: Path to the CSV file containing raw crack tip trajectory.
        """
        self.filepath = filepath
        self.window_length = None
        self.poly_order = None 
        self.tip = self._load_raw_trajectory()
    
    def set_savgol_params(self, window_length: int, poly_order: int):
        """
        Set the parameters for the Savitzky-Golay filter.
        :param window_length: Length of the filter window (must be an odd number).
        :param poly_order: Order of the polynomial used to fit the samples.
        """
        self.window_length = window_length
        self.poly_order = poly_order
    
    def _load_raw_trajectory(self):
        """
        Load the tip coordinates from the raw trajectory file.
        :return: DataFrame of tip coordinates and time.
        """
        raw_trajectory = pd.read_csv(self.filepath)
        return raw_trajectory
    
    def calc_smooth_trajectory(self, plot=False):
        """
        Smooth the crack tip trajectory using the Savitzky-Golay filter.
        :return: DataFrame of smoothed tip coordinates.
        """
        if self.window_length is None or self.poly_order is None:
            raise ValueError("Savitzky-Golay parameters not set. Use set_savgol_params to set them.")
        
        self.tip['Smoothed:0'] = savgol_filter(self.tip['Points:0'], self.window_length, self.poly_order)
        self.tip['Smoothed:1'] = savgol_filter(self.tip['Points:1'], self.window_length, self.poly_order)
        
        if plot:
            self.plot_trajectory(ax=None, smoothed=True)
            
        return self.tip[['Time', 'Smoothed:0', 'Smoothed:1']]
    
    def calc_crack_length(self, use_smoothed=False, plot=False):
        """
        Calculate the cumulative crack length over time.
        :param use_smoothed: Boolean indicating whether to use smoothed trajectory.
        :param plot: Boolean indicating whether to plot the results.
        :return: Series of cumulative crack lengths.
        """
        if use_smoothed:
            coords = ['Smoothed:0', 'Smoothed:1']
            length_col = 'Smoothed Length'
        else:
            coords = ['Points:0', 'Points:1']
            length_col = 'Raw Length'
        
        diff_coords = self.tip[coords].diff().fillna(0)
        diff_coords_squared = diff_coords.pow(2)
        sum_diff_coords_squared = diff_coords_squared.sum(axis=1)
        lengths = np.sqrt(sum_diff_coords_squared).cumsum()
        
        self.tip[length_col] = lengths
        
        if plot:
            self.plot_variable(length_col, 'Crack Length', use_smoothed)
        
        return self.tip[length_col]
    
    def calc_crack_speed(self, use_smoothed=False, plot=False):
        """
        Calculate the crack tip speed over time.
        :param use_smoothed: Boolean indicating whether to use smoothed trajectory.
        :param plot: Boolean indicating whether to plot the results.
        :return: Series of crack tip speeds.
        """
        if use_smoothed:
            length_col = 'Smoothed Length'
        else:
            length_col = 'Raw Length'

        times = self.tip['Time']
        speeds = self.tip[length_col].diff() / times.diff()
        speeds.iloc[0] = 0  # Set the first speed to 0 to match the original length of the input array
        
        speed_col = length_col.replace('Length', 'Speed')
        self.tip[speed_col] = speeds
        
        if plot:
            self.plot_variable(speed_col, 'Crack Speed', use_smoothed)
        
        return self.tip[speed_col]
    
    def calc_crack_velocity(self, direction, use_smoothed=False, plot=False):
        """
        Calculate the crack tip velocities in a specified direction over time.
        :param direction: Integer indicating velocity direction (x = 0 or y = 1)
        :param use_smoothed: Boolean indicating whether to plot the results.
        :return: Series of crack tip velocities in a specified coordinate direction.
        """
        if direction not in {0, 1}:
            raise ValueError("Parameter 'direction' must be 0 (x-direction or 1 (y-direction)")
        
        if use_smoothed:
            coords = ['Smoothed:0', 'Smoothed:1']
            length_cols = ['Smoothed Length:0', 'Smoothed Length:1']
        else:
            coords = ['Points:0', 'Points:1']
            length_cols = ['Raw Length:0', 'Raw Length:1']
        
        diff_coord = self.tip[coords[direction]].diff().fillna(0)
        diff_coord_squared = diff_coord.pow(2)
        lengths = np.sqrt(diff_coord_squared).cumsum()

        self.tip[length_cols[direction]] = lengths
        
        times = self.tip['Time']
        velocities = self.tip[length_cols[direction]].diff() / times.diff()
        velocities.iloc[0] = 0  # Set the first speed to 0 to match the original length of the input array
        
        velocity_col = length_cols[direction].replace('Length', 'Velocity')
        self.tip[velocity_col] = velocities

        if plot:
            direction_label = 'x' if direction == 0 else 'y'
            self.plot_variable(velocity_col, f'Crack Velocity in {direction_label}', use_smoothed)
        
        return self.tip[velocity_col]

    def plot_trajectory(self, ax=None, smoothed=False):
        """
        Plot the raw or smoothed crack tip trajectory.
        :param ax: The axes to plot on. If None, create a new figure.
        :param smoothed: Boolean indicating whether to plot smoothed trajectory.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if smoothed:
            x, y = self.tip['Smoothed:0'], self.tip['Smoothed:1']
            label = f"Smoothed Tip (window={self.window_length}, order={self.poly_order})"
            ax.plot(x, y, label=label)
        else:
            x, y = self.tip['Points:0'], self.tip['Points:1']
            label = "Raw Tip"
            ax.plot(x, y, 'k--', lw=0.8, label=label)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True)

    def plot_variable(self, variable_col, ylabel, use_smoothed, ax=None):
        """
        Plot a variable (length, velocity) against time.
        :param variable_col: Column name of the variable to plot.
        :param ylabel: Label for the y-axis.
        :param use_smoothed: Boolean indicating whether to use smoothed trajectory.
        :param ax: The axes to plot on. If None, create a new figure.
        """
        times = self.tip['Time']
        if ax is None:
            fig, ax = plt.subplots()
        
        label = ylabel
        if use_smoothed:
            label = f'Smoothed {ylabel} (window={self.window_length}, poly={self.poly_order})'
        else:
            label = f'Raw {ylabel}'
        ax.plot(times, self.tip[variable_col], label=label)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    def plot_crack_info(self, raw_lengths, smoothed_lengths, raw_speeds, smoothed_speeds, 
                        raw_velocities_x, raw_velocities_y, smoothed_velocities_x, smoothed_velocities_y):
        """
        Plot the raw and smoothed crack length, speed, and velocity information.
        :param raw_lengths: Series of raw cumulative crack lengths.
        :param smoothed_lengths: Series of smoothed cumulative crack lengths.
        :param raw_speeds: Series of raw crack tip speeds.
        :param smoothed_speeds: Series of smoothed crack tip speeds.
        :param raw_velocities_x: Series of raw crack tip velocities in x direction.
        :param smoothed_velocities_x: Series of smoothed crack tip velocities in x direction.
        :param raw_velocities_y: Series of raw crack tip velocities in y direction.
        :param smoothed_velocities_y: Series of smoothed crack tip velocities in y direction.
        """
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        # Plot trajectories
        self.plot_trajectory(axs[0, 0], smoothed=False)
        self.plot_trajectory(axs[0, 0], smoothed=True)
        axs[0, 0].set_title('Crack Tip Trajectory')

        # Plot lengths
        self.plot_variable('Raw Length', 'Crack Length', use_smoothed=False, ax=axs[0, 1])
        self.plot_variable('Smoothed Length', 'Crack Length', use_smoothed=True, ax=axs[0, 1])
        axs[0, 1].set_title('Crack Length over Time')

        # Plot velocities in x direction
        self.plot_variable('Raw Velocity:0', 'Crack Velocity in x', use_smoothed=False, ax=axs[1, 0])
        self.plot_variable('Smoothed Velocity:0', 'Crack Velocity in x', use_smoothed=True, ax=axs[1, 0])
        axs[1, 0].set_title('Crack Velocity in x over Time')

        # Plot velocities in y direction
        self.plot_variable('Raw Velocity:1', 'Crack Velocity in y', use_smoothed=False, ax=axs[1, 1])
        self.plot_variable('Smoothed Velocity:1', 'Crack Velocity in y', use_smoothed=True, ax=axs[1, 1])
        axs[1, 1].set_title('Crack Velocity in y over Time')

        # Plot speeds
        self.plot_variable('Raw Speed', 'Crack Speed', use_smoothed=False, ax=axs[2, 0])
        self.plot_variable('Smoothed Speed', 'Crack Speed', use_smoothed=True, ax=axs[2, 0])
        axs[2, 0].set_title('Crack Speed over Time')

        # Turn off bottom-right empty plot
        axs[2, 1].axis('off')
        axs[2, 1].set_facecolor('white')

        plt.tight_layout()
        plt.show()

    def analyze(self):
        """
        Perform the complete analysis: calculate raw and smoothed crack lengths, speed, and velocities,
        and plot the results.
        """
        raw_lengths = self.calc_crack_length(use_smoothed=False)
        raw_speeds = self.calc_crack_speed(use_smoothed=False)
        raw_velocities_x = self.calc_crack_velocity(0, use_smoothed=False)
        raw_velocities_y = self.calc_crack_velocity(1, use_smoothed=False)

        self.calc_smooth_trajectory()
        smoothed_lengths = self.calc_crack_length(use_smoothed=True)
        smoothed_speeds = self.calc_crack_speed(use_smoothed=True)
        smoothed_velocities_x = self.calc_crack_velocity(0, use_smoothed=True)
        smoothed_velocities_y = self.calc_crack_velocity(1, use_smoothed=True)

        self.plot_crack_info(raw_lengths, smoothed_lengths, raw_speeds, smoothed_speeds, 
                             raw_velocities_x, raw_velocities_y, smoothed_velocities_x, smoothed_velocities_y)

    def save_to_csv(self, output_filepath=None):
        """
        Save the updated DataFrame to a CSV file.
        :param output_filepath: Path to the output CSV file. If None, save to the original filepath.
        """
        if output_filepath is None:
            output_filepath = self.filepath
        self.tip.to_csv(output_filepath, index=False)