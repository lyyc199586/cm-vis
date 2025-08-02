"""
3D Grid and Voxel Data Processing
=================================

This subpackage provides utilities for processing and visualizing 3D grid
and voxel data, including loading from various formats and creating 3D
visualizations with proper edge detection and rendering.

Modules
-------
voxel : 3D voxel data processing and visualization

Functions
---------
load_vox : Load voxel data from text files
voxel_image : Render 3D voxel data as surface plots
voxel_contourf : Create contour plots on cube faces

Examples
--------
>>> from cm_vis.grid import load_vox, voxel_image
>>> import matplotlib.pyplot as plt
>>> 
>>> # Load and visualize voxel data
>>> data = load_vox("simulation.vox")
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> voxel_image(data, threshold=0.5, ax=ax)
"""

from .voxel import load_vox, voxel_image, voxel_contourf

__all__ = ['load_vox', 'voxel_image', 'voxel_contourf']