"""
3D Voxel Data Processing and Visualization
==========================================

This module provides utilities for loading, processing, and visualizing 3D voxel data.
It includes functions for loading voxel files, computing visible edges for 3D rendering,
and creating surface plots and contour visualizations of 3D scalar fields.

The module supports various visualization modes including surface rendering with
visible edge detection and multi-plane contour plotting for comprehensive 3D data
analysis.

Functions
---------
load_vox : Load voxel data from text files
find_furthest_corner : Determine furthest corner from camera
hidden_edges_from_furthest : Identify hidden edges for 3D rendering
plot_visible_edges : Plot visible bounding box edges
voxel_image : Render 3D voxel data as surface plot
voxel_contourf : Create contour plots on cube faces

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from cm_vis.grid import load_vox, voxel_image
>>> 
>>> # Load and visualize voxel data
>>> voxel_data = load_vox("data.vox")
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> voxel_image(voxel_data, threshold=0.5, ax=ax)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d
from typing import Optional, Tuple, Set, Dict
from pathlib import Path

def load_vox(path: Path) -> np.ndarray:
    """
    Load voxel data from a text file.
    
    Reads voxel data from a text file with format "x y z value" and
    constructs a 3D array where vox[x,y,z] contains the corresponding value.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to the voxel data file (typically .vox or .txt format)
        
    Returns
    -------
    numpy.ndarray
        3D array containing voxel values indexed by spatial coordinates
        
    Notes
    -----
    The input file should contain one line per voxel with format:
    x_coord y_coord z_coord value
    
    Coordinates can be separated by spaces, tabs, or commas.
    
    Examples
    --------
    >>> from pathlib import Path
    >>> voxel_data = load_vox(Path("damage.vox"))
    >>> print(voxel_data.shape)
    (100, 50, 25)
    """
    data = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=None, names=["x", "y", "z", "value"])  # shape: (N, 4)
    x_vals = np.sort(data["x"].unique())
    y_vals = np.sort(data["y"].unique())
    z_vals = np.sort(data["z"].unique())

    # build mesh
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    
    vox = data.pivot_table(index=["x", "y", "z"], values="value").values.reshape(len(x_vals), len(y_vals), len(z_vals))
    return vox


def find_furthest_corner(corners: Dict[str, Tuple[float, float, float]], ax: plt.Axes) -> str:
    """
    Find the corner of a cube that appears furthest from the camera.
    
    Determines which of the 8 cube corners has the largest projected
    z-coordinate (furthest from camera) in the current 3D view.
    
    Parameters
    ----------
    corners : dict of str to tuple
        Dictionary mapping corner labels (e.g., '000', '111') to their
        3D coordinates (x, y, z)
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D matplotlib axes object with current viewing transformation
        
    Returns
    -------
    str
        Label of the corner that appears furthest from the camera
        
    Notes
    -----
    Uses the current projection matrix to transform 3D coordinates to
    screen space and identifies the corner with maximum projected depth.
    
    Examples
    --------
    >>> corners = {'000': (0,0,0), '111': (1,1,1), ...}
    >>> furthest = find_furthest_corner(corners, ax)
    >>> print(f"Furthest corner: {furthest}")
    """
    max_z = -np.inf
    furthest_label = None
    proj = ax.get_proj()
    for label, (x, y, z) in corners.items():
        _, _, z_proj = proj3d.proj_transform(x, y, z, proj)
        if z_proj > max_z:
            max_z = z_proj
            furthest_label = label
    return furthest_label

def hidden_edges_from_furthest(corner: str) -> Set[Tuple[str, str]]:
    """
    Get the three edges connected to the furthest corner of a cube.
    
    For a given corner (specified as a 3-character binary string like '000'),
    returns the three edges that connect to that corner. These edges will
    typically be hidden from view when the corner is furthest from camera.
    
    Parameters
    ----------
    corner : str
        3-character string representing corner coordinates in binary
        (e.g., '000' for origin, '111' for opposite corner)
        
    Returns
    -------
    set of tuple of str
        Set containing three edge tuples, where each edge is represented
        as a sorted pair of corner labels
        
    Notes
    -----
    Corner labels use binary notation where each digit represents
    coordinates along x, y, z axes (0 = min, 1 = max).
    
    Examples
    --------
    >>> edges = hidden_edges_from_furthest('000')
    >>> print(edges)
    {('000', '001'), ('000', '010'), ('000', '100')}
    """
    def edge(a, b): return (a, b) if a < b else (b, a)

    x, y, z = corner

    edge_x = edge(corner, ('1' if x == '0' else '0') + y + z)
    edge_y = edge(corner, x + ('1' if y == '0' else '0') + z)
    edge_z = edge(corner, x + y + ('1' if z == '0' else '0'))

    return {edge_x, edge_y, edge_z}

def plot_visible_edges(
    ax: plt.Axes,
    bounds: Tuple[float, float, float, float, float, float],
    color: str = "k",
    zorder: float = 1e3,
    **kwargs
) -> None:
    """
    Plot only the visible edges of a 3D bounding box.
    
    Determines which edges of a cube/box are visible from the current
    viewing angle and plots only those edges to create a clean wireframe
    without hidden line removal artifacts.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D matplotlib axes object for plotting
    bounds : tuple of float
        Bounding box coordinates as (xmin, xmax, ymin, ymax, zmin, zmax)
    color : str, optional
        Color of the edge lines (default: 'k' for black)
    zorder : float, optional
        Drawing order for edge visibility (default: 1e3 for high priority)
    **kwargs
        Additional keyword arguments passed to plot3D for line styling
        
    Notes
    -----
    Uses view-dependent edge culling to show only the 9 visible edges
    of a cube's 12 total edges, providing a clean wireframe appearance.
    
    Examples
    --------
    >>> bounds = (0, 1, 0, 1, 0, 1)  # Unit cube
    >>> plot_visible_edges(ax, bounds, color='red', linewidth=2)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Define cube corners
    corners = {
        '000': (xmin, ymin, zmin),
        '100': (xmax, ymin, zmin),
        '010': (xmin, ymax, zmin),
        '110': (xmax, ymax, zmin),
        '001': (xmin, ymin, zmax),
        '101': (xmax, ymin, zmax),
        '011': (xmin, ymax, zmax),
        '111': (xmax, ymax, zmax),
    }

    # All 12 edges as (start_key, end_key)
    edges = [
        ('000', '100'), ('000', '010'), ('000', '001'),
        ('100', '110'), ('100', '101'),
        ('010', '110'), ('010', '011'),
        ('001', '101'), ('001', '011'),
        ('110', '111'),
        ('101', '111'),
        ('011', '111'),
    ]

    # Step 1: find most hidden corner
    furthest = find_furthest_corner(corners, ax)

    # Step 2: hide its 3 outgoing edges
    hidden = hidden_edges_from_furthest(furthest)

    # Plot visible edges
    edge_kw = dict(color=color, zorder=zorder)
    edge_kw.update(kwargs)
    for a_key, b_key in edges:
        if (a_key, b_key) in hidden or (b_key, a_key) in hidden:
            continue
        a, b = corners[a_key], corners[b_key]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **edge_kw)

def voxel_image(
    vox: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    shade: bool = False,
    edge: bool = True,
    edge_kw: Optional[dict] = None,
    surf_kw: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Render a 3D voxel scalar field as an isosurface plot.
    
    Creates a 3D surface visualization of voxel data by extracting and
    rendering the isosurface at the middle value of the data range.
    
    Parameters
    ----------
    vox : numpy.ndarray
        3D array representing the voxel scalar field
    vmin : float, optional
        Minimum value for colormap normalization (default: data minimum)
    vmax : float, optional
        Maximum value for colormap normalization (default: data maximum)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    shade : bool, optional
        Whether to apply surface shading (default: False)
    edge : bool, optional
        Whether to draw visible bounding box edges (default: True)
    edge_kw : dict, optional
        Additional keyword arguments for edge line styling (default: None)
    surf_kw : dict, optional
        Additional keyword arguments for surface plotting (default: None)
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        The 3D axes to plot on. If None, creates new figure (default: None)
        
    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object containing the plot
        
    Notes
    -----
    Uses marching cubes algorithm (via skimage) to extract isosurface
    at the median value between vmin and vmax.
    
    Examples
    --------
    >>> vox_data = load_vox("damage.vox")
    >>> ax = voxel_image(vox_data, cmap='plasma', edge=True)
    >>> ax.set_title("3D Damage Visualization")
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    if vmin is None:
        vmin = np.min(vox)
    if vmax is None:
        vmax = np.max(vox)
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    # meshgrid
    nx, ny, nz = vox.shape
    X, Y, Z = np.meshgrid(np.arange(nx+1), np.arange(ny+1), np.arange(nz+1))
    
    # face plotting map: (slicer, (X, Y, Z))
    face_slices = {
        "top":    (vox[:, :, -1],  X[:, :, -1], Y[:, :, -1], Z[:, :, -1]),
        "bottom": (vox[:, :,  0],  X[:, :,  0], Y[:, :,  0], Z[:, :,  0]),
        "front":  (vox[:, -1, :],  X[:, -1, :], Y[:, -1, :], Z[:, -1, :]),
        "back":   (vox[:,  0, :],  X[:,  0, :], Y[:,  0, :], Z[:,  0, :]),
        "right":  (vox[-1, :, :],  X[-1, :, :], Y[-1, :, :], Z[-1, :, :]),
        "left":   (vox[ 0, :, :],  X[ 0, :, :], Y[ 0, :, :], Z[ 0, :, :]),
    }
    
    # plot surfaces
    for face, (v, x, y, z) in face_slices.items():
        face_color = cmap(norm(v))
        default_surf_kw = dict(rstride=1, cstride=1, shade=shade)
        surf_kw = surf_kw or {}
        ax.plot_surface(x, y, z, 
                        facecolors=face_color,
                        **default_surf_kw,
                        **surf_kw)
    
    # bounding box limits
    xmin, xmax = 0, nx
    ymin, ymax = 0, ny
    zmin, zmax = 0, nz
    
    if edge:
        bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        edge_kw = edge_kw or {}
        plot_visible_edges(ax, bounds, **edge_kw)
    
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
    return ax

def voxel_contourf(
    vox: np.ndarray,
    level: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    edge: bool = True,
    edge_kw: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualize 3D voxel data using contour plots on all cube faces.
    
    Creates contour plots of the voxel scalar field on all six faces
    of the 3D bounding box, providing a comprehensive view of the data.
    
    Parameters
    ----------
    vox : numpy.ndarray
        3D array representing the voxel scalar field
    level : int, optional
        Number of contour levels to display (default: 10)
    vmin : float, optional
        Minimum value for colormap normalization (default: data minimum)
    vmax : float, optional
        Maximum value for colormap normalization (default: data maximum)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    edge : bool, optional
        Whether to draw visible bounding box edges (default: True)
    edge_kw : dict, optional
        Additional keyword arguments for edge line styling (default: None)
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        The 3D axes to plot on. If None, creates new figure (default: None)
        
    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object containing the contour plots
        
    Notes
    -----
    Displays contour plots on all six cube faces (front, back, top, bottom,
    left, right) using the same color scale for consistency.
    
    Examples
    --------
    >>> vox_data = load_vox("electric.vox")
    >>> ax = voxel_contourf(vox_data, level=15, cmap='coolwarm')
    >>> ax.set_title("Electric Field Distribution")
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if vmin is None:
        vmin = vox.min()
    if vmax is None:
        vmax = vox.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, level)
    kw = dict(levels=levels, norm=norm, extend="both", cmap=cmap)

    nx, ny, nz = vox.shape
    X, Y, Z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz)
    )

    # Define 6 cube faces: (data, x, y, z, zdir, offset)
    face_slices = [
        (vox[:, :, -1], X[:, :, -1], Y[:, :, -1], Z[:, :, -1], 'z', nz- 1), 
        (vox[:, :,  0], X[:, :,  0], Y[:, :,  0], Z[:, :,  0], 'z', 0),     
        (vox[:, -1, :], X[:, -1, :], Y[:, -1, :], Z[:, -1, :], 'x', nx - 1),
        (vox[:,  0, :], X[:,  0, :], Y[:,  0, :], Z[:,  0, :], 'x', 0),     
        (vox[-1, :, :], X[-1, :, :], Y[-1, :, :], Z[-1, :, :], 'y', ny - 1),
        (vox[ 0, :, :], X[ 0, :, :], Y[ 0, :, :], Z[ 0, :, :], 'y', 0),     
    ]

    for v, x, y, z, zdir, offset in face_slices:
        if zdir == 'z':
            ax.contourf(x, y, v, zdir=zdir, offset=offset, **kw)
        elif zdir == 'y':
            ax.contourf(x, v, z, zdir=zdir, offset=offset, **kw)
        else:  # zdir == 'x'
            ax.contourf(v, y, z, zdir=zdir, offset=offset, **kw)

    # Plot cube edge frame (optional)
    xmin, xmax = 0, nx-1
    ymin, ymax = 0, ny-1
    zmin, zmax = 0, nz-1
    
    if edge:
        bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        edge_kw = edge_kw or {}
        plot_visible_edges(ax, bounds, **edge_kw)

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
    return ax