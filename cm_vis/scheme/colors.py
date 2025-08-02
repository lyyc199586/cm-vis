"""
Custom Colormaps and Color Utilities
====================================

This module provides custom colormap functionality for scientific
visualization, including predefined colormaps and utilities for
creating custom color schemes with proper normalization.

Classes
-------
CMColormap : Custom colormap management class

Examples
--------
>>> from cm_vis.scheme import CMColormap
>>> import matplotlib.pyplot as plt
>>> 
>>> # Use predefined coolhot colormap
>>> cmap = CMColormap.get_coolhot(clim=[-1, 1])
>>> 
>>> # Create custom colormap
>>> custom_cmap = CMColormap(clim=[0, 100], colors=['blue', 'white', 'red'])
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


class CMColormap:
    """
    Custom colormap management for scientific visualization.
    
    This class provides functionality to create and manage custom colormaps
    with proper normalization and colorbar support. Includes predefined
    colormaps suitable for scientific data visualization.
    
    Parameters
    ----------
    clim : list, optional
        Color limits as [min_value, max_value] (default: [0, 1])
    colors : list of str, optional
        List of color names or hex codes to create colormap
    cmap : matplotlib.colors.Colormap, optional
        Existing matplotlib colormap to use
        
    Attributes
    ----------
    norm : matplotlib.colors.Normalize
        Normalization object for the colormap
    cmap : matplotlib.colors.Colormap
        The colormap object
    coolhot_cmap : matplotlib.colors.LinearSegmentedColormap
        Predefined coolhot colormap
        
    Class Attributes
    ----------------
    coolhot_colors : list
        RGB tuples defining the coolhot colormap
    coolhot_values : list
        Position values for coolhot colormap segments
        
    Examples
    --------
    >>> # Create custom colormap from colors
    >>> cmap = CMColormap(clim=[0, 10], colors=['blue', 'green', 'red'])
    >>> 
    >>> # Use matplotlib colormap
    >>> cmap = CMColormap(clim=[-5, 5], cmap=plt.cm.viridis)
    >>> 
    >>> # Get predefined coolhot colormap
    >>> coolhot = CMColormap.get_coolhot(clim=[0, 1])
    """

    # Predefined colormaps
    coolhot_colors = [(0, 1, 1), (0, 0, 1), (0, 0, 0.5), (1, 0, 0), (1, 1, 0)] 
    coolhot_values = [0.0, 0.45, 0.5, 0.55, 1]
    coolhot_cmap = LinearSegmentedColormap.from_list("coolhot", list(zip(coolhot_values, coolhot_colors)))

    def __init__(self, clim=[0, 1], colors=None, cmap=None):
        """
        Initialize custom colormap.
        
        Parameters
        ----------
        clim : list, optional
            Color limits as [min_value, max_value] (default: [0, 1])
        colors : list of str, optional
            Color names or hex codes to create linear colormap
        cmap : matplotlib.colors.Colormap, optional
            Existing matplotlib colormap to use
            
        Raises
        ------
        ValueError
            If neither colors nor cmap is provided
        """
        self.norm = Normalize(vmin=clim[0], vmax=clim[1])
        if cmap is None and colors is None:
            raise ValueError("Either colors of cmap should be provided!")

        if cmap is None:
            self.cmap = LinearSegmentedColormap.from_list("", colors)
        else:
            self.cmap = cmap

    def get_cmap(self):
        return self.cmap

    def get_norm(self):
        return self.norm

    def add_colorbar(self, cax, **kwargs):
        """to 'fake' a colorbar from CMColormap directly
        cax: axis where colorbar locate
        """
        cbar = ColorbarBase(cax, cmap=self.cmap, **kwargs)
        return cbar

    @classmethod
    def get_coolhot(cls, clim):
        """Get the predefined coolhot colormap with specified clim"""
        return cls(clim, cmap=cls.coolhot_cmap)
