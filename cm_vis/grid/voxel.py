"""
This module provides 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d
from typing import Optional, Tuple, Set
from pathlib import Path

def load_vox(path:Path) -> np.ndarray:
    """
    Load voxel data from a text file of format: x y z value
    
     Parameters:
    - path: path to vox.txt file

    Returns:
    - vox: 3D ndarray such that vox[x, y, z] = value
    """
    data = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=None, names=["x", "y", "z", "value"])  # shape: (N, 4)
    x_vals = np.sort(data["x"].unique())
    y_vals = np.sort(data["y"].unique())
    z_vals = np.sort(data["z"].unique())

    # build mesh
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    
    vox = data.pivot_table(index=["x", "y", "z"], values="value").values.reshape(len(x_vals), len(y_vals), len(z_vals))
    return vox


def find_furthest_corner(corners: dict, ax: plt.Axes) -> str:
    """
    Given 8 cube corners and ax, return the label of the furthest corner from the camera.
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
    Given a corner label like '000', return the 3 edges from that corner.
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
    Plot only the 9 visible edges of a voxel bounding box, based on view angle.

    Parameters:
    - ax: matplotlib 3D axes.
    - bounds: tuple of (xmin, xmax, ymin, ymax, zmin, zmax).
    - color: edge color (default: black).
    - zorder: drawing order (default: high to ensure visibility).
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
    edge_kw = dict(color="k", zorder=zorder)
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
    voxel: f(x, y, z) = value
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
    Visualize voxel scalar field using contourf on all 6 cube faces, and optional bounding edges.

    Parameters:
    - vox: ndarray (x, y, z)
    - level: number of contour levels
    - vmin/vmax: colormap range
    - cmap: colormap name
    - edge: whether to draw 9 visible bounding box edges
    - ax: optional 3D matplotlib axis
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
        (vox[:, :, -1], X[:, :, -1], Y[:, :, -1], Z[:, :, -1], 'z', nz- 1),  # top
        (vox[:, :,  0], X[:, :,  0], Y[:, :,  0], Z[:, :,  0], 'z', 0),          # bottom
        (vox[:, -1, :], X[:, -1, :], Y[:, -1, :], Z[:, -1, :], 'x', nx - 1),  # front
        (vox[:,  0, :], X[:,  0, :], Y[:,  0, :], Z[:,  0, :], 'x', 0),          # back
        (vox[-1, :, :], X[-1, :, :], Y[-1, :, :], Z[-1, :, :], 'y', ny - 1),  # right
        (vox[ 0, :, :], X[ 0, :, :], Y[ 0, :, :], Z[ 0, :, :], 'y', 0),          # left
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