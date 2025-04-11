"""
This module provides basic tools for creating 3D schemes in CM-VIS.
It includes classes for drawing 3D arrows, annotating dimensions, and adding boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
from matplotlib.patches import FancyArrowPatch, ArrowStyle, Circle, Polygon
from typing import List, Optional, Tuple

class Arrow3D(FancyArrowPatch):
    """
    A class to represent a 3D arrow in Matplotlib.
    Based on https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    """
    def __init__(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, *args, **kwargs):
        """
        Initialize a 3D arrow.

        Args:
            x0, y0, z0: Coordinates of the starting point.
            x1, y1, z1: Coordinates of the ending point.
            *args, **kwargs: Additional arguments for FancyArrowPatch.
        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self.xyz = [[x0, y0, z0], [x1, y1, z1]]
        
    def draw(self, renderer):
        """
        Draw the 3D arrow.

        Args:
            renderer: The renderer used for drawing.
        """
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None) -> float:
        """
        Perform the 3D projection of the arrow.

        Args:
            renderer: The renderer used for projection.

        Returns:
            The minimum z-coordinate of the arrow.
        """
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)

class Scheme3DBase:
    """
    A base class for creating 3D schemes with arrows, text, and coordinate axes.
    """
    def __init__(self, ax: plt.Axes, lw: float = 0.4):
        """
        Initialize the 3D scheme.

        Args:
            ax: The Matplotlib 3D axes.
            lw: Line width for drawing elements.
        """
        self.ax = ax
        self.lw = lw
        
    def add_arrow(self, type: str, xyz: List[List[float]], fc: Optional[str] = None):
        """
        Add a 3D arrow to the plot.

        Args:
            type: The arrow style type.
            xyz: Coordinates of the arrow as [[x0, y0, z0], [x1, y1, z1]].
            fc: Face color of the arrow.
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
        Add text annotation to the plot.

        Args:
            xyz: Coordinates of the text location.
            text: The text to display.
            offset: Offset for the text location as [dx, dy, dz].
            zdir: Direction of the text in 3D space.
            textc: Color of the text.
            boxfc: Background color of the text box.
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
        Draw coordinate axes at a specified origin.

        Args:
            origin: Origin of the coordinate axes.
            length: Length of each axis.
            text: Labels for the axes.
            textc: Color of the axis labels.
            shift: Factor to shift the labels along the axes.
        """
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = xyto[i] + length[i]
            self.add_arrow("-latex", xyz=(origin, xyto))
            self.add_text(np.array(xyto)*shift, text[i], textc=textc)

class Scheme3D(Scheme3DBase):
    """
    A class for creating advanced 3D schemes with annotations and boundary conditions.
    """
    def dim_dist(self, xyzfrom: List[float], xyzto: List[float], text: Optional[str] = None, 
                 zdir: Optional[str] = None, arrowfc: Optional[str] = None, 
                 textc: Optional[str] = None, boxfc: Optional[str] = None, offset: Optional[List[float]] = None):
        """
        Annotate a dimension with text in the center.

        Args:
            xyzfrom: Starting coordinates of the dimension.
            xyzto: Ending coordinates of the dimension.
            text: Text to display (default is the distance).
            zdir: Direction of the text in 3D space.
            arrowfc: Face color of the arrows.
            textc: Color of the text.
            boxfc: Background color of the text box.
            offset: Offset for the text location.
        """
        if(text is None):
            dist = np.sqrt((xyzfrom[0] - xyzto[0])**2 + (xyzfrom[1] - xyzto[1])**2) + (xyzfrom[2] - xyzto[2])**2
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