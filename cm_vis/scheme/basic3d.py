"""
3D Schematic Diagram Tools
==========================

This module provides tools for creating 3D schematic diagrams in CM-VIS.
It includes classes for drawing 3D arrows, annotating dimensions, and adding
boundary conditions to 3D matplotlib plots.

The module extends 2D schematic capabilities to three dimensions, supporting
3D arrows, dimensional annotations, and 3D boundary condition visualization
for engineering and scientific applications.

Classes
-------
Arrow3D : 3D arrow representation for matplotlib
Scheme3DBase : Base class for 3D schematic diagrams
Scheme3D : Extended 3D scheme class with annotations and boundary conditions

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from cm_vis.scheme import Scheme3D
>>> 
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> scheme3d = Scheme3D(ax)
>>> scheme3d.add_arrow("latex-latex", xyz=[[0, 0, 0], [1, 1, 1]])
>>> scheme3d.add_coord_axis()
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
from matplotlib.patches import FancyArrowPatch, ArrowStyle, Circle, Polygon
from typing import List, Optional, Tuple

class Arrow3D(FancyArrowPatch):
    """
    3D arrow representation for matplotlib plots.
    
    This class extends FancyArrowPatch to create 3D arrows that can be
    properly rendered in 3D matplotlib axes with correct projection
    and depth ordering.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the arrow starting point
    x1, y1, z1 : float
        Coordinates of the arrow ending point
    *args, **kwargs
        Additional arguments passed to FancyArrowPatch
        
    Attributes
    ----------
    xyz : list
        List containing start and end coordinates [[x0,y0,z0], [x1,y1,z1]]
        
    Notes
    -----
    Based on implementation from:
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    
    Examples
    --------
    >>> arrow = Arrow3D(0, 0, 0, 1, 1, 1, color='red', lw=2)
    >>> ax.add_artist(arrow)
    """
    def __init__(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, *args, **kwargs):
        """
        Initialize a 3D arrow.

        Parameters
        ----------
        x0, y0, z0 : float
            Coordinates of the starting point
        x1, y1, z1 : float
            Coordinates of the ending point
        *args, **kwargs
            Additional arguments for FancyArrowPatch
        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self.xyz = [[x0, y0, z0], [x1, y1, z1]]
        
    def draw(self, renderer):
        """
        Draw the 3D arrow with proper 2D projection.

        Parameters
        ----------
        renderer : matplotlib.backend_bases.RendererBase
            The renderer used for drawing
            
        Notes
        -----
        This method projects the 3D coordinates to 2D screen coordinates
        using the current 3D axes transformation matrix, then draws the
        arrow as a 2D patch.
        """
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None) -> float:
        """
        Perform 3D projection for depth ordering.

        Parameters
        ----------
        renderer : matplotlib.backend_bases.RendererBase, optional
            The renderer used for projection
            
        Returns
        -------
        float
            The minimum z-coordinate after projection, used for depth sorting
            
        Notes
        -----
        This method is called by matplotlib to determine drawing order
        for 3D objects. Objects with smaller return values are drawn first.
        """
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)

class Scheme3DBase:
    """
    Base class for creating 3D schematic diagrams.
    
    This class provides fundamental functionality for creating 3D schematic
    diagrams including arrows, text annotations, and coordinate axes.
    It serves as the foundation for more specialized 3D diagram classes.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The matplotlib 3D axes object for drawing
    lw : float, optional
        Default line width for drawing elements (default: 0.4)
        
    Attributes
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes for drawing
    lw : float
        Default line width
        
    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> scheme = Scheme3DBase(ax, lw=0.5)
    >>> scheme.add_arrow("-latex", xyz=[[0,0,0], [1,0,0]])
    """
    def __init__(self, ax: plt.Axes, lw: float = 0.4):
        """
        Initialize the 3D scheme base.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The matplotlib 3D axes object
        lw : float, optional
            Line width for drawing elements (default: 0.4)
        """
        self.ax = ax
        self.lw = lw
        
    def add_arrow(self, type: str, xyz: List[List[float]], fc: Optional[str] = None):
        """
        Add a 3D arrow to the diagram.
        
        Parameters
        ----------
        type : str
            Type of arrow head. Common types include:
            "->" : Simple arrow
            "-latex" : LaTeX-style arrow  
            "-stealth" : Stealth arrow
        xyz : list of two lists
            Start and end coordinates [[x0, y0, z0], [x1, y1, z1]]
        fc : str, optional
            Face color of the arrow head (default: None uses current color)
            
        Returns
        -------
        Arrow3D
            The created 3D arrow object
            
        Examples
        --------
        >>> coords = [[0, 0, 0], [1, 0, 0]]  # From origin to (1,0,0)
        >>> arrow = scheme.add_arrow("-latex", xyz=coords, fc='red')
        """
        if(fc is None):
            fc = 'k'
        
        arrow_props = dict(shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt", 
                      lw=self.lw, fc=fc, ec=fc)
        
        match type:
            case "-latex":
                style = ArrowStyle('-|>', head_length=6*self.lw, head_width=2*self.lw)
            case "latex-":
                style = ArrowStyle('<|-', head_length=6*self.lw, head_width=2*self.lw)
            case "latex-latex":
                style = ArrowStyle('<|-|>', head_length=6*self.lw, head_width=2*self.lw)
            case "-bar":
                style = ArrowStyle('|-|', widthA=0, widthB=4*self.lw)
            case "bar-":
                style = ArrowStyle('|-|', widthA=4*self.lw, widthB=0)
            case "bar-bar":
                style = ArrowStyle('|-|', widthA=4*self.lw, widthB=4*self.lw)
            case _:
                style = ArrowStyle('-')
        
        arrow_props.update(arrowstyle=style)
        
        # assign props and draw arrow
        (x0, y0, z0), (x1, y1, z1) = xyz
        arr = Arrow3D(x0, y0, z0, x1, y1, z1, **arrow_props)
        self.ax.add_artist(arr)
    
    def add_text(self, xyz: List[float], text: str, offset: Optional[List[float]] = None, 
                 zdir: Optional[str] = None, textc: Optional[str] = None, boxfc: Optional[str] = None):
        """
        Add 3D text annotation to the diagram.
        
        Parameters
        ----------
        xyz : list of float
            3D coordinates [x, y, z] for text placement
        text : str
            The text string to display
        offset : list of float, optional
            3D offset [dx, dy, dz] from the base coordinates (default: None)
        zdir : str, optional
            Direction for text orientation in 3D space.
            Options: 'x', 'y', 'z', None (default: None)
        textc : str, optional  
            Text color (default: None uses current color)
        boxfc : str, optional
            Background box face color (default: None for no box)
            
        Notes
        -----
        The text is positioned in 3D space and will be projected to 2D
        based on the current viewing angle.
        
        Examples
        --------
        >>> scheme.add_text([1, 0, 0], "Point A", offset=[0.1, 0, 0])
        >>> scheme.add_text([0, 1, 0], "Y-axis", zdir='y', textc='blue')
        """
        if(boxfc is None):
            boxfc = 'None'
            
        if(textc is None):
            textc = 'k'
            
        # get textx and texty
        x, y, z = xyz
        if(offset is not None):
            x = x + offset[0]
            y = y + offset[1]
            z = z + offset[2]
        
        self.ax.text(x, y, z, text, zdir=zdir,
                     bbox=dict(fc=boxfc, ec='none'), color=textc)
    
    def add_coord_axis(self, origin: List[float] = [0, 0, 0], length: List[float] = [1.0, 1.0, 1.0], 
                       text: List[str] = ['$x$', '$y$', '$z$'], textc: Optional[str] = None, shift: float = 1):
        """
        Draw 3D coordinate axes at a specified origin.
        
        Parameters
        ----------
        origin : list of float, optional
            Origin point [x0, y0, z0] for the coordinate system (default: [0, 0, 0])
        length : list of float, optional
            Length of each axis [lx, ly, lz] (default: [1.0, 1.0, 1.0])
        text : list of str, optional
            Labels for x, y, z axes (default: ['$x$', '$y$', '$z$'])
        textc : str, optional
            Color for axis labels (default: None uses current color)
        shift : float, optional
            Text label offset factor from axis endpoints (default: 1)
            
        Notes
        -----
        Creates three arrows representing the coordinate axes with LaTeX-style
        arrowheads and places text labels at the axis endpoints.
        
        Examples
        --------
        >>> # Standard coordinate system at origin
        >>> scheme.add_coord_axis()
        >>> 
        >>> # Custom coordinate system
        >>> scheme.add_coord_axis(origin=[1, 1, 1], length=[0.5, 0.5, 0.5],
        ...                      text=['X', 'Y', 'Z'], textc='red')
        """
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = xyto[i] + length[i]
            self.add_arrow("-latex", xyz=(origin, xyto))
            self.add_text(np.array(xyto)*shift, text[i], textc=textc)

class Scheme3D(Scheme3DBase):
    """
    Advanced 3D schematic diagram creator with specialized annotation tools.
    
    This class extends Scheme3DBase to provide specialized functionality for
    creating complex 3D diagrams including boundary condition indicators,
    stress/displacement annotations, and advanced geometric elements.
    
    Inherits all functionality from Scheme3DBase and adds specialized methods
    for engineering and scientific diagram creation.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The matplotlib 3D axes object for drawing
    lw : float, optional
        Default line width for drawing elements (default: 0.4)
        
    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> scheme = Scheme3D(ax)
    >>> scheme.add_coord_axis()
    >>> scheme.add_bc_indicator([1, 0, 0], "Fixed")
    """
    def dim_dist(self, xyzfrom: List[float], xyzto: List[float], text: Optional[str] = None, 
                 zdir: Optional[str] = None, arrowfc: Optional[str] = None, 
                 textc: Optional[str] = None, boxfc: Optional[str] = None, offset: Optional[List[float]] = None):
        """
        Create a dimension line with distance annotation between two points.
        
        Parameters
        ----------
        xyzfrom : list of float
            Starting coordinates [x1, y1, z1] of the dimension line
        xyzto : list of float
            Ending coordinates [x2, y2, z2] of the dimension line
        text : str, optional
            Custom text to display (default: None calculates distance automatically)
        zdir : str, optional
            Direction for text orientation ('x', 'y', 'z', None)
        arrowfc : str, optional
            Face color of dimension arrows (default: None uses current color)
        textc : str, optional
            Text color (default: None uses current color)
        boxfc : str, optional
            Background box color for text (default: None for no background)
        offset : list of float, optional
            3D offset [dx, dy, dz] for text position from line center
            
        Notes
        -----
        Creates a double-headed arrow between two points with the distance
        displayed at the center. If no custom text is provided, the
        Euclidean distance is calculated and displayed.
        
        Examples
        --------
        >>> # Dimension line with automatic distance calculation
        >>> scheme.dim_dist([0, 0, 0], [1, 0, 0])
        >>> 
        >>> # Custom dimension with text
        >>> scheme.dim_dist([0, 0, 0], [0, 2, 0], text="Height", 
        ...                textc='blue', arrowfc='red')
        """
        if(text is None):
            dist = np.sqrt((xyzfrom[0] - xyzto[0])**2 + (xyzfrom[1] - xyzto[1])**2 + (xyzfrom[2] - xyzto[2])**2)
            text = str(np.round(dist, 2))

        if(boxfc is None):
            boxfc = 'None'
        
        # add arrows
        self.add_arrow("latex-latex", xyz=(xyzfrom, xyzto), fc=arrowfc)
        
        # add text
        textx = (xyzfrom[0] + xyzto[0]) / 2
        texty = (xyzfrom[1] + xyzto[1]) / 2
        textz = (xyzfrom[2] + xyzto[2]) / 2
        self.add_text([textx, texty, textz], text, zdir=zdir, textc=textc, boxfc=boxfc, offset=offset)
    
    def draw_bc(self, type: str, xyz: np.ndarray, dir: np.ndarray, zdir: str):
        """
        Draw a boundary condition symbol.

        Args:
            type: Type of boundary condition ('fix' or 'roller').
            xyz: Coordinates of the boundary condition.
            dir: Direction vector for the boundary condition.
            zdir: Z direction in 3D space.
        """
        # get x, y and z based on zdir
        x, y, z = xyz + dir # shift shape center to the direction of dir
        match zdir:
            case 'x':
                xy1, z1 = (y, z), x
                d = np.array([dir[1], dir[2]])
            case 'y':
                xy1, z1 = (x, z), y
                d = np.array([dir[0], dir[2]])
            case 'z':
                xy1, z1 = (x, y), z
                d = np.array([dir[0], dir[1]])
            case _:
                raise ValueError(f'Unknown zdir: {zdir}')
        
        d_norm = np.linalg.norm(d)
        if d_norm == 0:
            raise ValueError("Direction vector `dir` cannot be zero in projection plane.")

        match type:
            case 'fix':
                # triangle: tip at xyz (boundary), base centered at xyz + dir
                tip = np.array(xy1) - d
                base_center = np.array(xy1)
                
                d_perp = np.array([-d[1], d[0]]) * 0.5
                base1 = base_center + d_perp
                base2 = base_center - d_perp
                p = Polygon([tip, base1, base2], closed=True, fc='w', ec='k', lw=self.lw)
                
            case 'roller':
                p = Circle(xy1, d_norm, fc='w', ec='k', lw=self.lw)
            case _:
                raise ValueError(f'Unknown boundary condition type: {type}')
            
        self.ax.add_patch(p)
        p.set_zorder(1000)
        pathpatch_2d_to_3d(p, z=z1, zdir=zdir)
            
    def add_fix_bc(self, bnd: np.ndarray, dir: List[float] = [1, 0, 0], zdir: str = 'z', spacing: int = 1):
        """
        Add fixed boundary conditions.

        Args:
            bnd: Array of boundary coordinates.
            dir: Direction vector for the boundary condition.
            zdir: Direction of the boundary condition in 3D space.
            spacing: Spacing between boundary condition symbols.
        """
        for s in np.array(bnd)[::spacing]:
            self.draw_bc('fix', s, dir=dir, zdir=zdir)
            
    def add_roller_bc(self, bnd: np.ndarray, dir: List[float] = [1, 0, 0], zdir: str = 'z', spacing: int = 1):
        """
        Add roller boundary conditions.

        Args:
            bnd: Array of boundary coordinates.
            dir: Direction vector for the boundary condition.
            zdir: Direction of the boundary condition in 3D space.
            spacing: Spacing between boundary condition symbols.
        """
        for s in np.array(bnd)[::spacing]:
            self.draw_bc('roller', s, dir=dir, zdir=zdir)
            
    def add_dist_bc(self, bnd: np.ndarray, bc: np.ndarray, type: str = "tail", scale: float = 1, spacing: int = 1):
        """
        Add distributed boundary conditions.

        Args:
            bnd: Array of boundary coordinates.
            bc: Array of boundary condition vectors.
            type: Type of distributed boundary condition ('tail' or 'head').
            scale: Scaling factor for the boundary condition vectors.
            spacing: Spacing between boundary condition symbols.
        """
        n_nodes = np.size(bnd, 0)
        if(type == "tail"):
            scale = scale
            arr_type = "-latex"
        else:
            scale = -scale
            arr_type = "latex-"
            
        bnd_s = np.vstack([bnd[:, 0] + scale*bc[:, 0],
                           bnd[:, 1] + scale*bc[:, 1],
                           bnd[:, 2] + scale*bc[:, 2]]).T
        
        for i in range(0, n_nodes, spacing):
            self.add_arrow(arr_type, xyz=(bnd[i], bnd_s[i]))
        
        self.ax.plot(bnd_s[::spacing, 0],
                     bnd_s[::spacing, 1],
                     bnd_s[::spacing, 2], color='k', lw=self.lw)