import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase

class CMColormap:
    '''custom color map class
    '''
    def __init__(self, clim, colors=None, cmap=None):
        '''initialize custom colormap
        colors = ["color1", "color2", ...] or
        cmap = cmap from matplotlib.cm
        clim = [cmin, cmax]
        '''
        self.norm = Normalize(vmin=clim[0], vmax=clim[1])
        if(cmap is None and colors is None):
            raise ValueError("Either colors of cmap should be provided!")
        
        if(cmap is None):
            self.cmap = LinearSegmentedColormap.from_list("", colors)
        else:
            self.cmap = cmap
        
    def get_cmap(self):
        return self.cmap
    
    def get_norm(self):
        return self.norm
    
    def add_colorbar(self, cax, **kwargs):
        ''' to 'fake' a colorbar from CMColormap directly
        cax: axis where colorbar locate
        '''
        cbar = ColorbarBase(cax, cmap=self.cmap, **kwargs)
        return cbar