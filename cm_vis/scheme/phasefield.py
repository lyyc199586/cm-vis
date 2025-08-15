"""
Phase Field Visualization Schemes
=================================

This module provides specialized schematic tools for phase field schematics.
It extends the basic scheme functionality to create phase field representations 
with crack paths and regularization zones.

Classes
-------
PFScheme : Phase field visualization scheme class

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from matplotlib.path import Path
>>> from cm_vis.scheme import PFScheme
>>> 
>>> fig, ax = plt.subplots()
>>> pf_scheme = PFScheme(ax)
>>> 
>>> # Create crack path
>>> vertices = np.array([[0, 0], [1, 0.5], [2, 0]])
>>> codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
>>> crack_path = Path(vertices, codes)
>>> 
>>> # Add phase field visualization
>>> pf_scheme.add_phasefield([crack_path], reg_length=0.1, cm=plt.cm.Reds)
"""

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from .basic import Scheme

class PFScheme(Scheme):
    """
    Phase field visualization scheme class.
    
    This class extends the basic Scheme class to provide specialized
    functionality for visualizing phase field models, particularly
    for crack propagation simulations. It can render phase fields
    with proper regularization zones and domain masking.
    
    Inherits all functionality from the base Scheme class and adds
    phase field specific visualization methods.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.path import Path
    >>> import matplotlib.cm as cm
    >>> 
    >>> fig, ax = plt.subplots()
    >>> pf = PFScheme(ax)
    >>> 
    >>> # Define crack geometry
    >>> crack_verts = np.array([[0, 0], [1, 0], [2, 0.5]])
    >>> crack_codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
    >>> crack = Path(crack_verts, crack_codes)
    >>> 
    >>> # Add phase field visualization
    >>> pf.add_phasefield([crack], reg_length=0.05, cm=cm.Reds)
    """
    
    def add_phasefield(self, crackset, reg_length, cm, domain=None):
        """
        Add phase field visualization from crack geometry.
        
        Creates a phase field representation by drawing cracks with varying
        line widths to simulate the regularization zone characteristic of
        phase field fracture models.
        
        Parameters
        ----------
        crackset : list of matplotlib.path.Path
            List of Path objects representing crack geometries
        reg_length : float
            Regularization length of the phase field model (controls the
            width of the diffuse crack zone)
        cm : matplotlib.colors.Colormap
            Colormap for the phase field visualization
        domain : matplotlib.path.Path, optional
            Domain boundary path for clipping the phase field visualization
            
        Returns
        -------
        matplotlib.path.Path
            Combined path of all cracks in the crackset
            
        Notes
        -----
        The phase field is rendered by drawing the crack paths with
        progressively smaller line widths and varying colors to create
        a diffuse zone effect. The outermost zone uses reg_length width
        while the innermost uses the default line width.
        
        Examples
        --------
        >>> # Simple straight crack
        >>> crack_path = Path(np.array([[0, 0], [2, 0]]), 
        ...                   [Path.MOVETO, Path.LINETO])
        >>> pf.add_phasefield([crack_path], reg_length=0.1, cm=plt.cm.coolwarm)
        >>> 
        >>> # Multiple cracks with domain clipping
        >>> domain_path = Path(square_vertices, square_codes)
        >>> pf.add_phasefield(crack_list, reg_length=0.05, 
        ...                   cm=plt.cm.Reds, domain=domain_path)
        """
        if(domain is not None):
            domain_patch = PathPatch(domain, fc='None', ec='None')
            self.ax.add_patch(domain_patch) # this is necessary for set_clip_path
            
        vertices = np.empty((0, 2))
        codes = []
        for crack_path in crackset:
            vertices = np.vstack((vertices, crack_path.vertices))
            codes.extend(crack_path.codes)
        
        path = Path(vertices, codes)
        
        # iteratively plot crackset of width to form phase field
        num = 20
        colors = cm(np.linspace(0, 1, num))
        for (i, width) in enumerate(np.linspace(reg_length, self.lw, num)):
            patch = PathPatch(path, lw=width, color=colors[i], fc='None',
                              capstyle='round')
            if(domain is not None):
                patch.set_clip_path(domain_patch)
            self.ax.add_patch(patch)
            
        if(domain is not None):
            # re-draw the outer boundary
            self.ax.plot(domain.vertices[:, 0], domain.vertices[:, 1], 'k', lw=self.lw)
        
        return path