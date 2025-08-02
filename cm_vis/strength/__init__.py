"""
Strength Surface Analysis Tools
==============================

This subpackage provides tools for generating, analyzing, and visualizing
material strength surfaces and failure criteria for various material models.

Modules
-------
gen : Strength surface generation for various failure criteria
plot : 3D visualization and plotting of strength surfaces

Classes
-------
StrengthSurface : Generate strength surface data
SurfacePlotter : Visualize and plot strength surfaces

Examples
--------
>>> from cm_vis.strength import StrengthSurface, SurfacePlotter
>>> 
>>> # Generate Von Mises surface
>>> surface = StrengthSurface("steel", "VMS", [250], [-400, 400, 101])
>>> data = surface.gen()
>>> 
>>> # Plot the surface
>>> plotter = SurfacePlotter("surface_data.npy")
>>> plotter.plot(option="3D")
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == 'StrengthSurface':
        from .gen import StrengthSurface
        return StrengthSurface
    elif name == 'SurfacePlotter':
        from .plot import SurfacePlotter
        return SurfacePlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .gen import StrengthSurface, SurfacePlotter
__all__ = ['StrengthSurface', 'SurfacePlotter']