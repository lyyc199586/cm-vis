"""
Crack Tip Trajectory Analysis
=============================

This module provides tools for analyzing crack tip trajectories from
simulation data. It includes functionality for smoothing trajectories,
calculating crack lengths and speeds, and visualizing crack propagation.

Classes
-------
CrackTipAnalyzer : Main class for crack tip trajectory analysis

Notes
-----
Uses Savitzky-Golay filtering for trajectory smoothing and supports
various crack propagation metrics calculation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class CrackTipAnalyzer:
    """
    Analyze crack tip trajectories from simulation data.
    
    This class provides methods to load, smooth, and analyze crack tip
    trajectory data. It supports calculation of crack lengths, speeds,
    and velocities with optional trajectory smoothing.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing raw crack tip trajectory data
        
    Attributes
    ----------
    filepath : str
        Path to the trajectory data file
    window_length : int or None
        Savitzky-Golay filter window length
    poly_order : int or None
        Savitzky-Golay filter polynomial order
    tip : pandas.DataFrame
        Loaded trajectory data
        
    Examples
    --------
    >>> # Load and analyze crack tip trajectory
    >>> analyzer = CrackTipAnalyzer("crack_trajectory.csv")
    >>> analyzer.set_savgol_params(window_length=11, poly_order=3)
    >>> 
    >>> # Calculate smoothed trajectory and crack length
    >>> smooth_traj = analyzer.calc_smooth_trajectory(plot=True)
    >>> crack_length = analyzer.calc_crack_length(use_smoothed=True, plot=True)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the CrackTipAnalyzer with trajectory data.
        
        Parameters
        ----------
        filepath : str
            Path to the CSV file containing raw crack tip trajectory data.
            Expected format: columns for Time, Points:0, Points:1
        """
        self.filepath = filepath
        self.window_length = None
        self.poly_order = None 
        self.tip = self._load_raw_trajectory()
    
    def set_savgol_params(self, window_length: int, poly_order: int):
        """
        Set parameters for the Savitzky-Golay filter.
        
        Parameters
        ----------
        window_length : int
            Length of the filter window (must be an odd number)
        poly_order : int
            Order of the polynomial used to fit the samples
            
        Raises
        ------
        ValueError
            If window_length is not odd or if poly_order >= window_length
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if poly_order >= window_length:
            raise ValueError("poly_order must be less than window_length")
            
        self.window_length = window_length
        self.poly_order = poly_order
    
    def _load_raw_trajectory(self):
        """
        Load tip coordinates from the raw trajectory file.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing time and tip coordinate data
        """
        raw_trajectory = pd.read_csv(self.filepath)
        return raw_trajectory
    
    def calc_smooth_trajectory(self, plot=False):
        """
        Smooth crack tip trajectory using Savitzky-Golay filter.
        
        Applies Savitzky-Golay filtering to both x and y coordinates of
        the crack tip trajectory to reduce noise while preserving features.
        
        Parameters
        ----------
        plot : bool, optional
            Whether to plot the smoothed trajectory (default: False)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with smoothed tip coordinates
            
        Raises
        ------
        ValueError
            If Savitzky-Golay parameters have not been set
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
        Calculate cumulative crack length over time.
        
        Computes the total crack length by integrating the incremental
        distances between consecutive crack tip positions.
        
        Parameters
        ----------
        use_smoothed : bool, optional
            Whether to use smoothed trajectory data (default: False)
        plot : bool, optional
            Whether to plot the crack length evolution (default: False)
            
        Returns
        -------
        pandas.Series
            Series of cumulative crack lengths over time
            
        Notes
        -----
        The crack length is calculated as the cumulative sum of Euclidean
        distances between consecutive crack tip positions.
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
        Calculate crack tip speed over time.
        
        Computes the instantaneous crack propagation speed as the time
        derivative of the crack length.
        
        Parameters
        ----------
        use_smoothed : bool, optional
            Whether to use smoothed trajectory data (default: False)
        plot : bool, optional
            Whether to plot the crack speed evolution (default: False)
            
        Returns
        -------
        pandas.Series
            Series of crack tip speeds over time
            
        Notes
        -----
        Speed is calculated as dL/dt where L is the crack length and t is time.
        The first speed value is set to 0.
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
        Plot the crack tip trajectory.
        
        Creates a 2D plot showing the path of the crack tip through space.
        Can display either raw or smoothed trajectory data.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure (default: None)
        smoothed : bool, optional
            Whether to plot smoothed trajectory data (default: False)
            
        Notes
        -----
        The plot includes appropriate labels, legend, and grid for clarity.
        Aspect ratio is set to equal for accurate spatial representation.
        
        Examples
        --------
        >>> # Plot raw trajectory
        >>> analyzer.plot_trajectory()
        >>> 
        >>> # Plot smoothed trajectory on existing axes
        >>> fig, ax = plt.subplots()
        >>> analyzer.plot_trajectory(ax=ax, smoothed=True)
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
        Plot a calculated variable against time.
        
        Creates a time series plot for variables such as crack length,
        speed, or velocity with appropriate styling and labels.
        
        Parameters
        ----------
        variable_col : str
            Column name of the variable to plot from the tip DataFrame
        ylabel : str
            Label for the y-axis
        use_smoothed : bool
            Whether the plotted data comes from smoothed calculations
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure (default: None)
            
        Notes
        -----
        The plot title indicates whether smoothed or raw data is shown.
        Grid is enabled for better readability of values.
        
        Examples
        --------
        >>> # Plot crack length evolution
        >>> analyzer.plot_variable('Raw Length', 'Crack Length', False)
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

    def plot_crack_info(self):
        """
        Plot comprehensive crack analysis information in a multi-panel figure.
        
        Creates a 3x2 subplot layout showing trajectory, lengths, speeds,
        and velocity components for both raw and smoothed data.
            
        Notes
        -----
        Creates a comprehensive visualization with 6 subplots showing:
        - Crack tip trajectory
        - Crack length evolution
        - Crack speed evolution  
        - Velocity components over time
        """
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

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

        self.plot_crack_info()

    def save_to_csv(self, output_filepath=None):
        """
        Save the updated DataFrame to a CSV file.
        :param output_filepath: Path to the output CSV file. If None, save to the original filepath.
        """
        if output_filepath is None:
            output_filepath = self.filepath
        self.tip.to_csv(output_filepath, index=False)