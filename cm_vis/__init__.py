"""
CM-VIS: Computational Mechanics Visualization Tools
===================================================

CM-VIS is a Python package for creating visualization schemes and plots 
for computational mechanics applications. It provides tools for:

- Creating schematic diagrams with boundary conditions and annotations
- Plotting finite element method (FEM) results
- Visualizing strength surfaces and material properties
- Processing and displaying 3D voxel data
- Creating flowcharts and technical diagrams

The package automatically loads custom matplotlib styles for consistent
scientific figure formatting.

Modules
-------
fem : Finite element analysis visualization tools
scheme : Schematic diagram creation tools  
strength : Strength surface calculation and plotting
grid : 3D voxel data processing and visualization
utils : Utility functions for axes and formatting

Notes
-----
Custom matplotlib styles are automatically loaded from the styles directory
and made available for use in plots.

Examples
--------
>>> # Import specific modules as needed
>>> from cm_vis.scheme import Scheme
>>> from cm_vis.utils import axis_line
>>> from cm_vis.grid import load_vox, voxel_image
"""

import os
from matplotlib.style.core import USER_LIBRARY_PATHS, reload_library

# get path to styles
styles_dir = os.path.join(os.path.dirname(__file__), 'styles')

# add styles to mpl
USER_LIBRARY_PATHS.append(styles_dir)
reload_library()

# Note: Import specific classes from submodules as needed to avoid dependency issues
# Example usage:
# from cm_vis.fem import FEMPlotter, Exodus, CrackTipAnalyzer, LDL
# from cm_vis.scheme import Scheme, Scheme3D, PFScheme, FlowScheme, CMColormap
# from cm_vis.strength import StrengthSurface, SurfacePlotter  
# from cm_vis.grid import load_vox, voxel_image, voxel_contourf
# from cm_vis.utils import axis_line

__version__ = "1.0.0"
__author__ = "Yangyuanchen Liu"
