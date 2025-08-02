"""
Strength Surface Visualization
==============================

This module provides tools for visualizing and plotting strength surfaces
in both 2D and 3D. It supports isosurface rendering, contour plotting,
and specialized views for plane stress and plane strain conditions.

Classes
-------
SurfacePlotter : Evaluate and plot strength surfaces

Examples
--------
>>> from cm_vis.strength import SurfacePlotter
>>> import matplotlib.pyplot as plt
>>> 
>>> # Load and plot 3D strength surface
>>> plotter = SurfacePlotter("strength_surface.npy")
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> plotter.plot("3D", ax=ax, save=True)
>>> 
>>> # Plot plane stress contours
>>> fig, ax = plt.subplots()
>>> plotter.plot("plane_stress", ax=ax, s3=0)
"""

# calculate and plot contour in 2d, isosurface in 3d from

import re
import numpy as np
import matplotlib.pyplot as plt
import s3dlib.surface as s3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, find_contours


class SurfacePlotter:
    """
    Evaluate and plot strength surfaces from generated data.
    
    This class provides methods to load strength surface data and create
    various visualizations including 3D isosurfaces, 2D contour plots,
    and specialized projections for plane stress/strain analysis.
    
    Parameters
    ----------
    data_dir : str
        Path to the .npy file containing strength surface data
        
    Attributes
    ----------
    dir : str
        Path to the data file
    srange : list
        Stress range extracted from filename
    surf : s3dlib.Surface3DCollection
        3D surface object for visualization
        
    Examples
    --------
    >>> # Basic 3D visualization
    >>> plotter = SurfacePlotter("surface_data.npy")
    >>> plotter.plot("3D")
    >>> 
    >>> # Plane stress analysis
    >>> plotter.plot("plane_stress", s3=0, save=True)
    >>> 
    >>> # Plane strain with custom Poisson's ratio
    >>> plotter.plot("plane_strain", nu=0.3, norm=100)
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize the surface plotter.
        
        Parameters
        ----------
        data_dir : str
            Path to the .npy file containing strength surface data
        """
        self.dir = data_dir
        self.srange = self.get_srange()
        self.surf = self.get_surf()
        
    def get_srange(self):
        """
        Extract stress range from filename.
        
        Parses the filename to extract stress range parameters that were
        used during surface generation.
        
        Returns
        -------
        list
            Stress range as [min_stress, max_stress, num_points]
        """
        pattern = r"srange\[((?:-?\d+(?:\.\d+)?(?:,\s*)?)+)\]"
        matches = re.findall(pattern, self.dir)
        if matches:
            srange = [float(val) for val in matches[0].split(",")]
        else:
            print("srange not found or wrong format!")
            srange = -1

        return srange
        
    def get_surf(self, norm: float = None, **kwargs):
        """
        Load and process surface data.
        
        Loads the strength surface data and creates a 3D surface object
        using marching cubes algorithm for isosurface extraction.
        
        Parameters
        ----------
        norm : float, optional
            Normalization factor for coordinates
        **kwargs
            Additional arguments passed to Surface3DCollection
            
        Returns
        -------
        s3dlib.Surface3DCollection
            3D surface object for visualization
        """
        # evaulate contour of isosurface
        xmin, xmax, num = self.srange
        num = int(num)
        dx = (xmax - xmin) / (num - 1)
        f = np.load(self.dir)

        f_smooth = gaussian_filter(f, sigma=1, order=0)
        verts, faces, _, _ = marching_cubes(f, level=0)

        # relocate vertices
        verts[:, 0] = (verts[:, 0] * dx + xmin) / (norm if norm else 1)
        verts[:, 1] = (verts[:, 1] * dx + xmin) / (norm if norm else 1)
        verts[:, 2] = (verts[:, 2] * dx + xmin) / (norm if norm else 1)
        
        self.surf = s3d.Surface3DCollection(verts, faces, **kwargs)
        return self.surf

    def plot(self, option: str = "3D", ax=None, save: bool = False, norm: float = None, nu: float = None, s3: float = 0, **kwargs):
        """
        Plot the strength surface with various visualization options.
        
        Creates visualizations of the strength surface including 3D isosurfaces,
        2D contour plots for plane stress/strain conditions, with optional
        data export functionality.
        
        Parameters
        ----------
        option : str, optional
            Visualization type: "3D", "plane_stress", or "plane_strain" (default: "3D")
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes for plotting. If None, creates new axes
        save : bool, optional
            Whether to save plot data to CSV files (default: False)
        norm : float, optional
            Normalization factor for coordinates and labels
        nu : float, optional
            Poisson's ratio for plane strain calculation (required for plane_strain)
        s3 : float, optional
            Value for s3 in plane stress condition (default: 0)
        **kwargs
            Additional arguments passed to plotting functions
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
            
        Notes
        -----
        For plane strain analysis, nu (Poisson's ratio) must be provided.
        The plane strain condition assumes s3 = nu * (s1 + s2).
        
        When save=True, the following files are created:
        - For 3D: *_3d_verts.csv and *_3d_faces.csv
        - For 2D: *_2d_contour.csv
        
        Examples
        --------
        >>> # 3D visualization with normalization
        >>> ax = plotter.plot("3D", norm=100, save=True)
        >>> 
        >>> # Plane stress contour plot
        >>> ax = plotter.plot("plane_stress", s3=0)
        >>> 
        >>> # Plane strain analysis
        >>> ax = plotter.plot("plane_strain", nu=0.3)
        """

        # evaulate contour of isosurface
        xmin, xmax, num = self.srange
        num = int(num)
        dx = (xmax - xmin) / (num - 1)
        f = np.load(self.dir)
        
        if option == "3D":

            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_proj_type("ortho")
                if norm is not None:
                    ax.set_xlabel("norm(s1)")
                    ax.set_ylabel("norm(s2)")
                    ax.set_zlabel("norm(s3)")
                else:
                    ax.set_xlabel("s1")
                    ax.set_ylabel("s2")
                    ax.set_zlabel("s3")

            # load surf
            ax.add_collection3d(self.surf.shade())
            ax.set(xlim=[xmin, xmax], ylim=[xmin, xmax], zlim=[xmin, xmax])
            ax.set_title(str(self.surf))

            if save:
                # Get vertices and faces from the surface object
                f_temp = np.load(self.dir)
                verts_temp, faces_temp, _, _ = marching_cubes(f_temp, level=0)
                
                # Apply same coordinate transformation as in get_surf
                verts_temp[:, 0] = verts_temp[:, 0] * dx + xmin
                verts_temp[:, 1] = verts_temp[:, 1] * dx + xmin  
                verts_temp[:, 2] = verts_temp[:, 2] * dx + xmin
                
                if norm is not None:
                    verts_temp = verts_temp / norm
                
                verts_dir = self.dir.replace(".npy", "_3d_verts.csv")
                faces_dir = self.dir.replace(".npy", "_3d_faces.csv")
                np.savetxt(verts_dir, verts_temp, delimiter=",", fmt="%.2f")
                np.savetxt(faces_dir, faces_temp, delimiter=",", fmt="%d")

        elif option == "plane_strain":
            # Plane strain 2D plot (s1, s2)
            s1_range = np.linspace(xmin, xmax, num)
            s2_range = np.linspace(xmin, xmax, num)
            s1_grid, s2_grid = np.meshgrid(s1_range, s2_range)

            # Calculate s3 = nu * (s1 + s2)
            s3_grid = nu * (s1_grid + s2_grid)
            s3_indices = np.round((s3_grid - xmin) / dx).astype(int)

            # Ensure the indices are within bounds
            s3_indices = np.clip(s3_indices, 0, num - 1)

            # Extract the plane strain surface using advanced indexing
            plane_strain_surface = f[np.arange(num)[:, None], np.arange(num), s3_indices]

            # Apply Gaussian smoothing to the plane strain surface
            plane_strain_surface_smooth = gaussian_filter(plane_strain_surface, sigma=1)

            # Extract the contour of the plane strain surface
            contours = find_contours(plane_strain_surface_smooth, 0)

            if ax is None:
                fig, ax = plt.subplots()
                ax.set_aspect("equal")
                if norm is not None:
                    ax.set_xlabel("norm(s1")
                    ax.set_ylabel("norm(s2)")
                else:
                    ax.set_xlabel("s1")
                    ax.set_ylabel("s2")

            for contour in contours:
                ax.plot((contour[:, 0] * dx + xmin) / (norm if norm else 1), (contour[:, 1] * dx + xmin) / (norm if norm else 1), **kwargs)

            if save:
                save_dir_2d = self.dir.replace(".npy", f"2d_plane_strain_nu{nu}.csv")
                np.savetxt(save_dir_2d, np.column_stack((contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin)), delimiter=",", fmt="%.2f")

        elif option == "plane_stress":
            # Plane stress cross-section at s3 = s3 (default is 0)
            z_index = int((s3 - xmin) / dx)  # index at z=0 plane, for plane stress
            contours = find_contours(f[:, :, z_index], 0)

            if ax is None:
                fig, ax = plt.subplots()
                ax.set_aspect("equal")
                if norm is not None:
                    ax.set_xlabel("norm(s11)")
                    ax.set_ylabel("norm(s22)")
                else:
                    ax.set_xlabel("s11")
                    ax.set_ylabel("s22")

            for contour in contours:
                ax.plot((contour[:, 0] * dx + xmin) / (norm if norm else 1), (contour[:, 1] * dx + xmin) / (norm if norm else 1), **kwargs)

            if save:
                save_dir = self.dir.replace(".npy", f"2d_plane_stress_s3{s3}.csv")
                np.savetxt(save_dir, np.column_stack((contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin)), delimiter=",", fmt="%.2f")

        return ax
