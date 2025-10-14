"""
Utility Functions
=================

This subpackage provides utility functions for enhancing matplotlib plots
and scientific visualizations, including custom axis formatting and
plot styling tools.

Modules
-------
axis : Custom axis line utilities for scientific plots

Functions
---------
axis_line : Create custom axis lines with arrows and origins

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from cm_vis.utils import axis_line
>>> 
>>> fig, ax = plt.subplots()
>>> ax.plot([-2, 2], [1, 4])
>>> axis_line(ax, origin=[0, 0], labels=['X', 'Y'])
"""

from .axis import axis_line
from .image import crop
from .fig import figsize, to_inch, lock_canvas

__all__ = [
    'axis_line',
    'crop',
    'figsize',
    'to_inch',
    'lock_canvas',
]