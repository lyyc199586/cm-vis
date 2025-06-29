"""
This module provides classes and methods for creating and annotating 
schematic diagrams in CM-VIS. It includes functionality for 
adding arrows, text, coordinate axes, paths, and boundary conditions.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.patheffects as patheffects
from matplotlib.path import Path
from matplotlib.patches import Arc, FancyArrowPatch, ArrowStyle, PathPatch, Polygon
from typing import List, Optional, Tuple, Union, Literal

class SchemeBase:
    """
    Base class for creating schematic diagrams with matplotlib.
    Provides methods for adding arrows, text, coordinate axes, and paths.
    """
    
    def __init__(self, ax, lw: float = 0.4) -> None:
        """
        Initialize the SchemeBase object.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object where the diagram will be drawn.
            lw (float): Line width for drawing elements.
        """
        self.ax = ax
        self.x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
        self.y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
        self.max_len = max(self.x_len, self.y_len)
        self.lw = lw
    
    def add_arrow(
        self, 
        type: str, 
        xy: Optional[List[List[float]]] = None, 
        path: Optional[Path] = None, 
        fc: Optional[str] = None,
        ec: Optional[str] = None,
        hw: Optional[float] = None,
        hl: Optional[float] = None,
        tw: Optional[float] = None
    ) -> None:
        """
        Add an arrow to the diagram.

        Args:
            type (str): Arrow style (e.g., "-latex", "latex-latex").
            xy (Optional[List[List[float]]]): Coordinates for a straight arrow, [[x0, y0], [x1, y1]].
            path (Optional[Path]): Path object for a curved arrow.
            fc (Optional[str]): Face color of the arrow.
            ec (Optional[str]): Edge color of the arrow.
            hw (Optional[float]): Head width of the arrow.
            hl (Optional[float]): Head length of the arrow.
            tw (Optional[float]): Tail width of the arrow.
        """
        if(xy is None and path is None ):
            raise ValueError("Provide either xy=(xyfrom, xyto) or path!")
        
        if(fc is None):
            fc = 'k'
            
        if(ec is None):
            ec = 'k'
            
        if(hw is None):
            hw = 2*self.lw
            
        if(hl is None):
            hl = 6*self.lw
            
        if(tw is None):
            tw = 2*self.lw
            
                
        if(path is None):
            arrow_props = dict(posA=xy[0], posB=xy[1], shrinkA=0, shrinkB=0, lw=self.lw,
                           joinstyle="miter", capstyle="butt", fc=fc, ec=ec)
        else:
            arrow_props = dict(path=path, shrinkA=0, shrinkB=0, lw=self.lw,
                           joinstyle="miter", capstyle="butt", fc=fc, ec=ec)
        
        match type:
            case "-latex":
                style = ArrowStyle('-|>', head_length=hl, head_width=hw)
            case "latex-":
                style = ArrowStyle('<|-', head_length=hl, head_width=hw)
            case "latex-latex":
                style = ArrowStyle('<|-|>', head_length=hl, head_width=hw)
            case "-bar":
                style = ArrowStyle('|-|', widthA=0, widthB=2*hw)
            case "bar-":
                style = ArrowStyle('|-|', widthA=2*hw, widthB=0)
            case "bar-bar":
                style = ArrowStyle('|-|', widthA=2*hw, widthB=2*hw)
            case "simple":
                style = ArrowStyle('simple', head_length=hl, head_width=4*hw, tail_width=2*tw)
            case "fancy":
                style = ArrowStyle('fancy', head_length=hl, head_width=4*hw, tail_width=2*tw)
            case "wedge":
                style = ArrowStyle('wedge', tail_width=2*tw)
            case _:
                style = ArrowStyle('-')
        
        # assign props and draw arrow
        arrow_props.update(arrowstyle=style)
        arr = FancyArrowPatch(**arrow_props)
        self.ax.add_patch(arr)
        
    def add_path_arrow(
        self,
        points: List[Tuple[float, float]],
        type: str = "-latex",
        fc: Optional[str] = None,
        ec: Optional[str] = None
    ) -> None:
        """
        Draw an arrow along a polyline defined by a sequence of points.

        Args:
            points: List of (x, y) points defining the path (at least 2).
            type: Arrow style string.
            fc: Face color.
            ec: Edge color.
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points to draw a path arrow.")

        codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
        path = Path(points, codes)
        self.add_arrow(type=type, path=path, fc=fc, ec=ec)
    
    def add_text(
        self, 
        textx: float, 
        texty: float, 
        text: str, 
        fs: Union[None, float, Literal['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']] = None,
        textc: Optional[str] = None, 
        boxfc: Optional[str] = None, 
        boxec: Optional[str] = None,
        loc: Optional[str] = None, 
        rotation: Optional[float] = None,
        offset: Optional[float] = None    ) -> None:
        """
        Add text to the diagram.

        Args:
            textx (float): X-coordinate of the text.
            texty (float): Y-coordinate of the text.
            text (str): Text content.
            fs (Union[None, float, Literal]): Font size of the text.
            textc (Optional[str]): Text color.
            boxfc (Optional[str]): Background color of the text box.
            boxec (Optional[str]): Edge color of the text box.
            loc (Optional[str]): Location of the text relative to the coordinates.
            rotation (Optional[float]): Rotation angle of the text.
            offset (Optional[float]): Offset for positioning the text.
        """
        
        textloc = None
        if(boxfc is None):
            boxfc = 'None'
            
        if(boxec is None):
            boxec = 'None'
            
        if(textc is None):
            textc = 'k'
        
        if(offset is None):
            offset = 0.01
            
        match loc:
            case "center":
                textloc = dict(ha='center', va='center')
            case "upper":
                textloc = dict(ha='center', va='bottom')
                texty = texty + offset*self.max_len
            case "lower":
                textloc = dict(ha='center', va='top')
                texty = texty - offset*self.max_len
            case "left":
                textloc = dict(ha='right', va='center')
                textx = textx - offset*self.max_len
            case "right":
                textloc = dict(ha='left', va='center')
                textx = textx + offset*self.max_len
            case _:
                textloc = dict(ha='center', va='bottom')
        self.ax.text(textx, texty, text, textloc, fontsize=fs, rotation=rotation,
                     bbox=dict(fc=boxfc, ec=boxec), color=textc)
        
    def add_coord_axis(
        self, 
        origin: List[float] = [0.0, 0.0], 
        length: List[float] = [1.0, 1.0], 
        text: List[str] = ['$x$', '$y$'], 
        textlocs: List[str] = ["right", "upper"], 
        textc: Optional[str] = None, 
        arrowfc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Draw coordinate axes at a specified origin.

        Args:
            origin (List[float]): Origin of the coordinate axes.
            length (List[float]): Length of the axes.
            text (List[str]): Labels for the axes.
            textlocs (List[str]): Locations of the axis labels.
            textc (Optional[str]): Text color for the labels.
            arrowfc (Optional[str]): Face color for the arrows.
            offset (Optional[float]): Offset for the labels.
        """
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = origin[i] + length[i]
            self.add_arrow("-latex", xy=(origin, xyto), fc=arrowfc)
            self.add_text(xyto[0], xyto[1], text[i], loc=textlocs[i], 
                          textc=textc, offset=offset)
    
    def add_3d_axis(
        self,
        origin : List[float] = [0.0, 0.0],
        x_dir : List[float] = [1.0, 0.0],
        y_dir : List[float] = [0.0, 1.0],
        z_dir : List[float] = [1.0, 1.0],
        text: List[str] = ['$x$', '$y$', '$z$'], 
        textlocs: List[str] = ["right", "upper", "upper"], 
        textc: Optional[str] = None, 
        arrowfc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Draw a pseudo 3d coordinate axes at a specified origin
        """
        if textc is None:
            textc = ['k', 'k', 'k']
        
        for axis, t, tloc, tcolor  in zip([x_dir, y_dir, z_dir], text, textlocs, textc):
            xyfrom = origin.copy()
            xyto = np.array(axis) + np.array(xyfrom)
            self.add_arrow("-latex", xy=(origin, xyto), fc=arrowfc)
            self.add_text(xyto[0], xyto[1], t, loc=tloc, 
                          textc=tcolor, offset=offset)
        
            
    def add_pathpatch(
        self, 
        verts: np.ndarray, 
        curve: bool = True, 
        closed: bool = False, 
        draw: bool = True, 
        **kwargs
    ) -> Path:
        """
        Generate a path of lines or curves passing through vertices.

        Args:
            verts (np.ndarray): Array of vertices [[x0, y0], ...].
            curve (bool): Whether to draw a smooth curve.
            closed (bool): Whether to close the path.
            draw (bool): Whether to draw the path on the axes.
            **kwargs: Additional arguments for the PathPatch.

        Returns:
            Path: Path object representing the generated path.
        """
        if(closed):
            verts = np.vstack((verts, verts[0]))
        
        if(curve):
            # if curve, do interpolation
            t = np.arange(np.size(verts, 0))
            if(np.size(verts, 0) < 3):
                raise ValueError("At least 4 points are needed to generate curve!")
            
            ti = np.linspace(0, t.max(), 10 * t.size)
            xi = interp1d(t, verts[:, 0], kind="cubic")(ti)
            yi = interp1d(t, verts[:, 1], kind="cubic")(ti)
            points = np.vstack((xi, yi)).T
            codes = [Path.MOVETO] + [Path.LINETO] * (np.size(points, 0) - 1)
            if(closed):
                codes[-1] = Path.CLOSEPOLY
            
        else:
            # if line, simply connect them
            points = verts
            codes = [Path.MOVETO] + [Path.LINETO] * (np.size(points, 0) - 1)
            if(closed):
                codes[-1] = Path.CLOSEPOLY
        
        path = Path(points, codes)
        if(curve):
            path = path.interpolated(steps=10)
        
        if("lw" not in kwargs):
            patch = PathPatch(path, lw=self.lw, **kwargs)
        else:
            patch = PathPatch(path, **kwargs)
        
        if(draw):
            self.ax.add_patch(patch)
        return path
    
class Scheme(SchemeBase):
    """
    Extended class for creating schematic diagrams with additional 
    methods for annotating dimensions, radii, angles, and boundary conditions.
    """
    
    def dim_dist(
        self, 
        xyfrom: List[float], 
        xyto: List[float], 
        text: Optional[str] = None, 
        arrowfc: Optional[str] = None, 
        textc: Optional[str] = None, 
        boxfc: Optional[str] = None, 
        textloc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Annotate a dimension with text in the center.

        Args:
            xyfrom (List[float]): Starting point of the dimension.
            xyto (List[float]): Ending point of the dimension.
            text (Optional[str]): Text to annotate the dimension.
            arrowfc (Optional[str]): Face color for the arrows.
            textc (Optional[str]): Text color.
            boxfc (Optional[str]): Background color of the text box.
            textloc (Optional[str]): Location of the text.
            offset (Optional[float]): Offset for positioning the text.
        """
        if(text is None):
            dist = np.sqrt((xyfrom[0] - xyto[0])**2 + (xyfrom[1] - xyto[1])**2)
            text = str(np.round(dist, 2))

        if(boxfc is None):
            boxfc = 'None'

        # add arrows
        self.add_arrow("latex-latex", xy=(xyfrom, xyto), fc=arrowfc)
        self.add_arrow("bar-bar", xy=(xyfrom, xyto), fc=arrowfc)
        
        # add text
        textx = (xyfrom[0] + xyto[0])/2
        texty = (xyfrom[1] + xyto[1])/2
        self.add_text(textx, texty, text, textc=textc, boxfc=boxfc, loc=textloc, offset=offset)

    def dim_radius(
        self, 
        center: List[float], 
        radius: float, 
        angle: float = 45, 
        arrowfc: Optional[str] = None, 
        text: Optional[str] = None, 
        textc: Optional[str] = None, 
        boxfc: Optional[str] = None, 
        textloc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Annotate a radius for an arc.

        Args:
            center (List[float]): Center of the arc.
            radius (float): Radius of the arc.
            angle (float): Angle of the radius line in degrees.
            arrowfc (Optional[str]): Face color for the arrow.
            text (Optional[str]): Text to annotate the radius.
            textc (Optional[str]): Text color.
            boxfc (Optional[str]): Background color of the text box.
            textloc (Optional[str]): Location of the text.
            offset (Optional[float]): Offset for positioning the text.
        """
        if(text is None):
            text = str(radius)

        xyto = [center[0] + radius*np.cos(angle/180*np.pi), 
                center[1] + radius*np.sin(angle/180*np.pi)]
        self.add_arrow("-latex", xy=(center, xyto), fc=arrowfc)
        self.add_text(xyto[0]/2, xyto[1]/2, text, textc, boxfc, loc=textloc, offset=offset)
        
    def dim_angle(
        self, 
        radius: float, 
        start_deg: float, 
        stop_deg: float, 
        xyfrom: Optional[List[float]] = None, 
        center: Optional[List[float]] = None, 
        arrowloc: str = "stop", 
        arrowfc: Optional[str] = None, 
        text: Optional[str] = None, 
        textc: Optional[str] = None, 
        textloc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Annotate an angle for an arc.

        Args:
            radius (float): Radius of the arc.
            start_deg (float): Starting angle in degrees.
            stop_deg (float): Ending angle in degrees.
            xyfrom (Optional[List[float]]): Starting point of the arc.
            center (Optional[List[float]]): Center of the arc.
            arrowloc (str): Location of the arrow ("stop", "start", "both", or "None").
            arrowfc (Optional[str]): Face color for the arrow.
            text (Optional[str]): Text to annotate the angle.
            textc (Optional[str]): Text color.
            textloc (Optional[str]): Location of the text.
            offset (Optional[float]): Offset for positioning the text.
        """
        if((xyfrom is None and center is None) or (xyfrom is None and center is None)):
            raise ValueError("Provide either xyfrom=[x0, y0] or center=[x0, y0]!")
        
        start = np.radians(start_deg)
        stop = np.radians(stop_deg)
        
        if(xyfrom is not None):
            center = [xyfrom[0] - radius*np.cos(start), xyfrom[1] - radius*np.sin(start)]
        else:
            xyfrom = [center[0] + radius*np.cos(start), center[1] + radius*np.sin(start)]
        
        
        if(text is None):
            text = str(abs(stop_deg - start_deg))
        
        xyto = [center[0] + radius*np.cos(stop), center[1] + radius*np.sin(stop)]
        arc = Arc(center, 2*radius, 2*radius, angle=0, theta1=start_deg, theta2=stop_deg)
        arc_path = arc.get_patch_transform().transform_path(arc.get_path())
        
        # generate arrow based on arc_path
        self.add_arrow("bar-bar", path=arc_path, fc=arrowfc)
        match arrowloc:
            case "stop":
                self.add_arrow("-latex", path=arc_path, fc=arrowfc)
            case "start":
                self.add_arrow("latex-", path=arc_path, fc=arrowfc)
            case "both":
                self.add_arrow("latex-latex", path=arc_path, fc=arrowfc)
            case _:
                self.add_arrow(path=arc_path, fc=arrowfc)
        
        # add text
        textxy = [center[0] + radius*np.cos((start + stop)/2), 
                  center[1] + radius*np.sin((start + stop)/2)]
        self.add_text(textxy[0], textxy[1], text, textc, loc=textloc, offset=offset)
    
    def add_fix_bc(
        self, 
        bnd: np.ndarray, 
        scale: float = 1, 
        spacing: float = 1, 
        angle: float = 45, 
        **kwargs
    ) -> None:
        """
        Annotate a fixed boundary with short inclined lines.

        Args:
            bnd (np.ndarray): Boundary nodes as an array of coordinates.
            scale (float): Scale factor for the line length.
            spacing (float): Spacing between the ticks.
            angle (float): Angle between the path and the tick.
            **kwargs: Additional arguments for the plot.
        """
        length = scale*4*self.lw
        self.ax.plot(bnd[:, 0], bnd[:, 1], c='k',
                     path_effects=[patheffects.withTickedStroke(spacing=spacing,
                                                                angle=angle,
                                                                length=length)], **kwargs)
        
    def add_point_bc(
        self, 
        bnd: np.ndarray, 
        bc: np.ndarray, 
        type: str = "tail", 
        scale: float = 1, 
        arrowfc: Optional[str] = None, 
        text: Optional[str] = None, 
        textc: Optional[str] = None, 
        textloc: Optional[str] = None, 
        offset: Optional[float] = None
    ) -> None:
        """
        Annotate a point boundary condition with an arrow.

        Args:
            bnd (np.ndarray): Boundary node as a coordinate array.
            bc (np.ndarray): Boundary condition as a displacement vector.
            type (str): Arrow type ("tail" or "head").
            scale (float): Scale factor for the arrow length.
            arrowfc (Optional[str]): Face color for the arrow.
            text (Optional[str]): Text to annotate the boundary condition.
            textc (Optional[str]): Text color.
            textloc (Optional[str]): Location of the text.
            offset (Optional[float]): Offset for positioning the text.
        """
        if(type == "tail"):
            scale = scale
            bnd_s = bnd + scale*bc
            self.add_arrow("-latex", xy=(bnd, bnd_s), fc=arrowfc)
            self.add_arrow("bar-", xy=(bnd, bnd_s), fc=arrowfc)
        else:
            scale = -scale
            bnd_s = bnd + scale*bc
            self.add_arrow("latex-", xy=(bnd, bnd_s), fc=arrowfc)
            self.add_arrow("-bar", xy=(bnd, bnd_s), fc=arrowfc)
        
        if(text is not None):
            self.add_text((bnd[0] + bnd_s[0])/2, (bnd[1] + bnd_s[1])/2, 
                          text=text, textc=textc, loc=textloc, offset=offset)
        
    def add_dist_bc(
        self, 
        bnd: np.ndarray, 
        bc: np.ndarray, 
        type: str = "tail", 
        scale: float = 1, 
        interval: int = 1, 
        arrowfc: Optional[str] = None, 
        text: Optional[str] = None, 
        textc: Optional[str] = None, 
        textloc: Optional[str] = None, 
        offset: Optional[float] = None, 
        **kwargs
    ) -> None:
        """
        Annotate a distributed boundary condition with arrows.

        Args:
            bnd (np.ndarray): Boundary nodes as an array of coordinates.
            bc (np.ndarray): Boundary conditions as an array of displacement vectors.
            type (str): Arrow type ("tail" or "head").
            scale (float): Scale factor for the arrow length.
            interval (int): Interval between arrows.
            arrowfc (Optional[str]): Face color for the arrows.
            text (Optional[str]): Text to annotate the boundary condition.
            textc (Optional[str]): Text color.
            textloc (Optional[str]): Location of the text.
            offset (Optional[float]): Offset for positioning the text.
            **kwargs: Additional arguments for the plot.
        """
        n_nodes = np.size(bnd, 0)
        if(type == "tail"):
            scale = scale
            arr_type = "-latex"
        else:
            scale = -scale
            arr_type = "latex-"
            
        bnd_s = np.vstack([bnd[:, 0] + scale*bc[:, 0], bnd[:, 1] + scale*bc[:, 1]]).T
            
        for i in range(0, n_nodes, interval):
            self.add_arrow(arr_type, xy=(bnd[i], bnd_s[i]), fc=arrowfc)
        
        self.ax.plot(bnd_s[::interval, 0], bnd_s[::interval, 1], 'k', **kwargs)
        
        if(text is not None):
            self.add_text(bnd_s[n_nodes//2, 0], bnd_s[n_nodes//2, 1], 
                          text=text, textc=textc, loc=textloc, offset=offset)
            
    
    def add_normal_bc(self) -> None:
        """
        Annotate a normal boundary condition (to be implemented).
        """
        # TODO: add_normal_bc based on add_dist_bc, normal to the boundary
        pass
    
    def add_cube(
        self,
        origin: List[float] = [0.0, 0.0],
        size: float = 1.0,
        depth: float = 0.2,
        text: Optional[str] = None,
        ec: Optional[str] = None,
        front_color: Optional[str] = None,
        side_color: Optional[str] = None,
        top_color: Optional[str] = None,
        **text_props: dict
    ) -> None:
        """
        Draw a pseudo-3D cube to the diagram.

        Args:
            origin (List[float]): Bottom left of the cube.
            size (float): Size of the cube.
            depth (float): Depth of the cube.
            text (Optional[str]): Text to annotate the cube.
            ec (Optional[str]): Edge color of the cube.
            front_color (Optional[str]): Face color of the cube.
            side_color (Optional[str]): Side color of the cube.
            top_color (Optional[str]): Top color of the cube.
        """
        x0, y0 = origin
        dx = size
        offset = depth * dx 
        if(ec is None):
            ec = 'k'
        if(front_color is None):
            front_color = 'white'
        if(side_color is None):
            side_color = 'lightgray'
        if(top_color is None):
            top_color = 'white'
        
        # Front face
        front = [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dx), (x0, y0 + dx)]
        # Top face
        top = [(x0, y0 + dx), (x0 + dx, y0 + dx),
               (x0 + dx + offset, y0 + dx + offset), (x0 + offset, y0 + dx + offset)]
        # Side face
        side = [(x0 + dx, y0), (x0 + dx + offset, y0 + offset),
                (x0 + dx + offset, y0 + dx + offset), (x0 + dx, y0 + dx)]

        # Draw the faces
        style = {'edgecolor': ec, 'lw': self.lw, 'joinstyle': 'bevel'}
        self.ax.add_patch(Polygon(front, closed=True, facecolor=front_color, **style))
        self.ax.add_patch(Polygon(top, closed=True, facecolor=top_color, **style))
        self.ax.add_patch(Polygon(side, closed=True, facecolor=side_color, **style))
        
        if text is not None:
            # Calculate the position for the text
            text_x = x0 + dx / 2
            text_y = y0 + dx / 2
            self.add_text(text_x, text_y, text, loc='center', **text_props)