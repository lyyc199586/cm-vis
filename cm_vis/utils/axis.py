"""
Axis Utilities for Scientific Plots
===================================

This module provides utility functions for customizing matplotlib axes
to create scientific and engineering style plots with custom axis lines,
origins, and arrow-style axis indicators.

Functions
---------
axis_line : Create custom axis lines with arrows and custom origins

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from cm_vis.utils import axis_line
>>> 
>>> fig, ax = plt.subplots()
>>> ax.plot([-2, 2], [1, 4])
>>> axis_line(ax, origin=[0, 0], labels=['X', 'Y'])
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, ArrowStyle

def axis_line(ax, lw=0.4, origin=[0, 0], labels=None, arrowfc='k', offset=0.05):
    """
    Create custom axis lines with arrows intersecting at specified origin.
    
    This function modifies a matplotlib axes object to display custom axis
    lines with arrow heads that intersect at a specified point, commonly
    used in scientific and engineering plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify
    lw : float, optional
        Line width of the arrow axes (default: 0.4)
    origin : list of float, optional
        The [x, y] coordinates of the axis intersection point (default: [0, 0])
    labels : list of str, optional
        Axis labels as [x_label, y_label]. If None, no labels are added
    arrowfc : str, optional
        Face color of the arrow heads (default: 'k' for black)
    offset : float, optional
        Ratio to extend the axis limits beyond data range (0.0-1.0, default: 0.05)
        
    Notes
    -----
    This function hides the default matplotlib spines and ticks, replacing them
    with custom arrow-style axes. The axis limits are automatically adjusted
    to ensure the origin point is visible with the specified offset.
    
    Examples
    --------
    >>> # Create a plot with custom axes at origin
    >>> fig, ax = plt.subplots()
    >>> ax.plot([-1, 3], [-2, 2])
    >>> axis_line(ax, origin=[0, 0], labels=['Distance (m)', 'Force (N)'])
    >>> 
    >>> # Custom styling with different origin
    >>> axis_line(ax, lw=0.6, origin=[1, 1], arrowfc='red', offset=0.1)
    """
    # Set the spines to be at the specified position
    ax.spines['left'].set_position(('data', origin[0]))
    ax.spines['bottom'].set_position(('data', origin[1]))
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    # Move the ticks to the bottom and left
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate the extension based on the current axis limits
    x_ext = (xlim[1] - xlim[0]) * offset
    y_ext = (ylim[1] - ylim[0]) * offset
    
    # Adjust axis limits to include the origin point
    xlim = (min(xlim[0], origin[0]) - x_ext, max(xlim[1], origin[0]) + x_ext)
    ylim = (min(ylim[0], origin[1]) - y_ext, max(ylim[1], origin[1]) + y_ext)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    xy_from = [[min(xlim[0], origin[0]), origin[1]], [origin[0], min(ylim[0], origin[1])]]
    xy_to = [[max(xlim[1], origin[0]), origin[1]], [origin[0], max(ylim[1], origin[1])]]
    
    # Create arrows
    for i in range(2):
        start = xy_from[i]
        end = xy_to[i]
        arrow_props = dict(posA=start, posB=end, shrinkA=0, shrinkB=0, lw=lw,
                           joinstyle="miter", capstyle="butt", fc=arrowfc, ec=arrowfc)
        style = ArrowStyle('-|>', head_length=6*lw, head_width=2*lw)
        arrow_props.update(arrowstyle=style)
        arr = FancyArrowPatch(**arrow_props)
        ax.add_patch(arr)
    
    # set labels
    if labels is not None:
        x_length = xlim[1] - xlim[0]
        y_length = ylim[1] - ylim[0]
        ax.set_xlabel(labels[0])
        ax.xaxis.set_label_coords(1.05, (origin[1] - ylim[0])/y_length)
        ax.set_ylabel(labels[1], rotation=0)
        ax.yaxis.set_label_coords((origin[0] - xlim[0])/x_length, 1.05)