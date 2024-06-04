import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


class CMColormap:
    """custom color map class"""

    # Predefined colormaps
    coolhot_colors = [(0, 1, 1), (0, 0, 1), (0, 0, 0.5), (1, 0, 0), (1, 1, 0)] 
    coolhot_values = [0.0, 0.45, 0.5, 0.55, 1]
    coolhot_cmap = LinearSegmentedColormap.from_list("coolhot", list(zip(coolhot_values, coolhot_colors)))

    def __init__(self, clim=[0, 1], colors=None, cmap=None):
        """initialize custom colormap
        colors = ["color1", "color2", ...] or
        cmap = cmap from matplotlib.cm
        clim = [cmin, cmax]
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
