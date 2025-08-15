"""
FEM Results Visualization
=========================

This module provides the FEMPlotter class for visualizing finite element
analysis results on 2D meshes. It supports plotting nodal and elemental
variables on triangular and quadrilateral elements with proper scaling
and color mapping.

Classes
-------
FEMPlotter : Main class for plotting FEM variables on mesh

Notes
-----
Currently optimized for 2D visualization. Requires mesh data in netCDF format
compatible with MOOSE/Exodus format.
"""

# use matplotlib to do simple plots with FEM mesh and variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection


class FEMPlotter:
    """
    Plot variable fields on finite element meshes (currently 2D only).
    
    This class provides methods to visualize nodal and elemental variables
    on finite element meshes with proper interpolation and color mapping.
    Supports both triangular and quadrilateral elements.
    
    Parameters
    ----------
    model : object
        NetCDF dataset containing mesh and variable information
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes for plotting. If None, creates new axes.
        
    Attributes
    ----------
    model : object
        Reference to the mesh/data model
    ax : matplotlib.axes.Axes
        Plotting axes
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from cm_vis.fem import Exodus, FEMPlotter
    >>> 
    >>> # Load exodus file and create plotter
    >>> exo = Exodus("simulation.e")
    >>> fig, ax = plt.subplots()
    >>> plotter = FEMPlotter(exo, ax)
    >>> 
    >>> # Plot temperature field at time step 10
    >>> plotter.plot("temperature", tstep=10, cmap="coolwarm")
    """

    def __init__(self, model, ax=None) -> None:
        """
        Initialize the FEM plotter.
        
        Parameters
        ----------
        model : object
            NetCDF dataset containing mesh and field data
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes for plotting. If None, generates new axes.
        """
        self.model = model
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.ax.set(aspect="equal")

    def plot(self, var=None, tstep=0, block_id=None, clim=None, cmap=None, scale=None, **kwargs):
        """
        Plot a variable field on the finite element mesh.
        
        This method visualizes nodal or elemental variables on the mesh with
        appropriate interpolation and color mapping. Supports both single block
        and multi-block plotting.
        
        Parameters
        ----------
        var : str or np.ndarray, optional
            Variable name (str) or variable data (ndarray) to plot.
            If None, plots mesh wireframe only.
        tstep : int, optional
            Time step index for variable extraction (default: 0)
        block_id : int, optional
            Element block ID. If None, plots all blocks (default: None)
        clim : tuple, optional
            Color limits as (vmin, vmax). If None, uses data range
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for visualization. If None, uses 'coolwarm'
        scale : list, optional
            Mesh coordinate scaling as [scale_x, scale_y, scale_z]
        **kwargs
            Additional arguments passed to matplotlib plotting functions
            
        Returns
        -------
        tuple
            - If block_id is None: (ax, [patch_list]) for all blocks
            - If block_id specified: (ax, patch) for single block
            
        Examples
        --------
        >>> # Plot temperature field on all blocks
        >>> ax, patches = plotter.plot("temperature", tstep=10)
        >>> 
        >>> # Plot stress on specific block with custom colormap
        >>> ax, patch = plotter.plot("stress", block_id=1, cmap="plasma")
        >>> 
        >>> # Plot mesh wireframe only
        >>> ax, patch = plotter.plot()
        """

        def quad_to_tri(quad):
            """convert a quad into 2 traiangles"""
            tris = [quad[:3], [quad[0], quad[2], quad[3]]]
            return tris

        def plot_block(
            self, block_id=None, tstep=0, var=None, clim=None, cmap=None, scale=scale, **kwargs
        ):
            """this is used to plot at a single block"""
            verts, faces = self.model.get_mesh(block_id=block_id, tstep=tstep)
            if scale is not None:
                for i in range(np.size(verts, 1)):
                    verts[:, i] = verts[:, i] * scale[i]

            n_nodes = np.size(verts, 0)
            n_elements = np.size(faces, 0)
            n_nodes_of_element = np.size(faces, 1)

            # only plot mesh if no variable provided
            if var is None:
                p = PolyCollection(
                    [verts[face] for face in faces],
                    closed=True,
                    antialiaseds=True,
                    facecolor="None",
                    edgecolor="k",
                    **kwargs
                )
                self.ax.add_collection(p)
                return (self.ax, p)
            else:
                if clim is None:
                    clim = (np.min(var), np.max(var))
                if cmap is None:
                    cmap = plt.cm.coolwarm

            # check variable type
            if np.size(var, 0) == n_nodes:
                var_type = "nodal"
            elif np.size(var, 0) == n_elements:
                var_type = "elemental"
            else:
                raise ValueError("Unsupported variable type!")

            if var_type == "nodal":
                if n_nodes_of_element == 3:
                    # triangle mesh
                    triangles = faces
                else:
                    tris = []
                    for quad in faces:
                        triangle = quad_to_tri(quad)
                        tris.extend(triangle)
                    triangles = np.array(tris)

                # plot nodal value with tripcolor
                tri_obj = tri.Triangulation(verts[:, 0], verts[:, 1], triangles)
                p = self.ax.tripcolor(
                    tri_obj, var, clim=clim, cmap=cmap, shading="gouraud", **kwargs
                )

            else:
                # plot elemental variable with PolyCollection
                p = PolyCollection(
                    [verts[face] for face in faces],
                    closed=True,
                    array=var,
                    cmap=cmap,
                    antialiaseds=True,
                    edgecolor="face",
                    **kwargs
                )
                p.set_clim(clim)
                self.ax.add_collection(p)

            return (self.ax, p)

        if block_id is None:
            p_lists = []
            block_nums, _ = self.model.get_block_info()
            if block_nums == 1:
                _, p = plot_block(self, 0, tstep, var, clim, cmap, **kwargs)
                self.ax.autoscale(enable=True)
                return (self.ax, p)
            else:
                for i in range(block_nums):
                    _, p = plot_block(self, i, tstep, var, clim, cmap, **kwargs)
                    p_lists.append(p)
                self.ax.autoscale(enable=True)
                return (self.ax, p_lists)
        else:
            _, p = plot_block(self, block_id, tstep, var, clim, cmap, **kwargs)
            self.ax.autoscale(enable=True)
            return (self.ax, p)
