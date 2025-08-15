"""
Schematic Diagram Creation Tools
===============================

This subpackage provides comprehensive tools for creating technical schematic
diagrams, flowcharts, and scientific illustrations with matplotlib.

Modules
-------
basic : Base classes for 2D schematic diagrams
basic3d : 3D schematic diagram tools
phasefield : Phase field model visualization
flow : Flowchart creation tools
colors : Custom colormaps and color utilities

Classes
-------
SchemeBase : Base class for 2D schemes
Scheme : Extended 2D scheme class with annotations
Scheme3D : 3D schematic diagram class
PFScheme : Phase field visualization scheme
FlowScheme : Flowchart management class
Node : Base flowchart node class
Cube : 3D cube flowchart node
CMColormap : Custom colormap utilities
"""

# cm_vis/scheme/__init__.py
from .basic import SchemeBase, Scheme
from .basic3d import Scheme3D
from .phasefield import PFScheme
from .flow import FlowScheme, Node, Cube
from .colors import CMColormap

__all__ = ['SchemeBase', 'Scheme', 'Scheme3D', 'PFScheme', 'FlowScheme', 'Node', 'Cube', 'CMColormap']