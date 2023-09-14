import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.patheffects as patheffects
from matplotlib.path import Path
from matplotlib.patches import Arc, FancyArrowPatch, ArrowStyle, PathPatch
from matplotlib.collections import LineCollection, PatchCollection

class SchemeBase:
    
    def __init__(self, ax, lw=None) -> None:
        self.ax = ax
        self.x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
        self.y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
        self.max_len = max(self.x_len, self.y_len)
        if(lw is None):
            self.lw = 0.4
        else:
            self.lw = lw
    
    def add_arrow(self, type, xy=None, path=None):
        '''base function to draw straight arraw:
        type: customized matplotlib arrow type,
        use posA=xy[0], posB=xy[1] for straight path, or path=path for any path
        '''
        if(xy is None and path is None ):
            raise ValueError("Provide either xy=(xyfrom, xyto) or path!")
        
        if(path is None):
            arrow_props = dict(posA=xy[0], posB=xy[1], shrinkA=0, shrinkB=0, lw=self.lw,
                           joinstyle="miter", capstyle="butt", fc='k')
        else:
            arrow_props = dict(path=path, shrinkA=0, shrinkB=0, lw=self.lw,
                           joinstyle="miter", capstyle="butt", fc='k')
        
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
        
        # assign props and draw arrow
        arrow_props.update(arrowstyle=style)
        arr = FancyArrowPatch(**arrow_props)
        self.ax.add_patch(arr)
    
    def add_text(self, textx, texty, text, textfc=None, loc=None, offset=None):
        '''base function to draw text, a wrapper of ax.text
        '''
        
        textloc = None
        if(textfc is None):
            textfc = 'None'
        
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
        self.ax.text(textx, texty, text, textloc, bbox=dict(fc=textfc, ec='none'))
        
    def add_coord_axis(self, origin=np.array([0.0, 0.0]), length=np.array([1.0, 1.0]), 
                       text=['$x$', '$y$'], offset=None):
        '''draw coordinates at origin
        '''
        textloc = ["right", "upper"]
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = xyto[i] + length[i]
            self.add_arrow("-latex", xy=(origin, xyto))
            self.add_text(xyto[0], xyto[1], text[i], loc=textloc[i], offset=offset)
            
    def add_pathpatch(self, verts, curve=True, closed=False, draw=True, **kwargs):
        '''base funciton to generate path of lines or curves passing vertices
        verts: np.array([[x0, y0], ...])
        if curve=Ture, uses scipy.interpolate to draw a smooth curve, need at least 4 vertices
        return path
        '''
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
    
    def dim_dist(self, xyfrom, xyto, text=None, textfc=None, textloc=None, offset=None):
        '''annotate dimension with text in the center:
        |<|--- text ---|>|, if text is None, use the distance
        ax: matplotlib axes object
        xyfrom: [x0, y0]
        xyto: [x1, y1]
        '''
        if(text is None):
            dist = np.sqrt((xyfrom[0] - xyto[0])**2 + (xyfrom[1] - xyto[1])**2)
            text = str(np.round(dist, 2))

        if(textfc is None):
            textfc = 'None'

        # add arrows
        self.add_arrow("latex-latex", xy=(xyfrom, xyto))
        self.add_arrow("bar-bar", xy=(xyfrom, xyto))
        
        # add text
        textx = (xyfrom[0] + xyto[0])/2
        texty = (xyfrom[1] + xyto[1])/2
        self.add_text(textx, texty, text, textfc, loc=textloc, offset=offset)

    def dim_radius(self, center, radius, angle=45, text=None, textfc=None, textloc=None, offset=None):
        '''annotate radius for arc:
        --text--|>, if text is None, use str(radius)
        '''
        if(text is None):
            text = str(radius)

        xyto = [center[0] + radius*np.cos(angle/180*np.pi), 
                center[1] + radius*np.sin(angle/180*np.pi)]
        self.add_arrow("-latex", xy=(center, xyto))
        self.add_text(xyto[0]/2, xyto[1]/2, text, textfc, loc=textloc, offset=offset)
        
    def dim_angle(self, radius, start_deg, stop_deg, xyfrom=None, center=None, 
                  arrowloc="stop", text=None, textloc=None, offset=None):
        '''annotate angle for arc: just like \draw(x, y) arc (start_deg:stop_deg:radius)
        |<--text-->|, if text is None, use str(abs(stop_deg - start_deg))
        provide either xyfrom=[x0, y0] or center=[x0, y0]
        arrowloc: "stop", "start", "both" or "None"
        '''
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
        self.add_arrow("bar-bar", path=arc_path)
        match arrowloc:
            case "stop":
                self.add_arrow("-latex", path=arc_path)
            case "start":
                self.add_arrow("latex-", path=arc_path)
            case "both":
                self.add_arrow("latex-latex", path=arc_path)
            case _:
                self.add_arrow(path=arc_path)
        
        # add text
        textxy = [center[0] + radius*np.cos((start + stop)/2), 
                  center[1] + radius*np.sin((start + stop)/2)]
        self.add_text(textxy[0], textxy[1], text, loc=textloc, offset=offset)
    
    def add_fix_bc(self, bnd, scale=1, spacing=1, angle=45):
        '''annotate fix boundary with short inclined lines: ///////
        use matplotlib.patheffects.withTickedStroke
        bnd: boundary nodes, dim: n_nodes*2, [[x1, y1], ..., [xn, yn]]
        loc: "upper", "lower", "left", "right"
        scale: scale the length of the arrow
        spacing: spacing between ticks
        angle: the angle between the path and the tick
        '''
        length = scale*4*self.lw
        self.ax.plot(bnd[:, 0], bnd[:, 1], c='k',
                     path_effects=[patheffects.withTickedStroke(spacing=spacing,
                                                                angle=angle,
                                                                length=length)])
        
    def add_point_bc(self, bnd, bc, type="tail", scale=1, text=None, textloc=None, offset=None):
        '''base function to annotate point boundary condition: |-->
        bnd: boundary node, np.array([x1, y1])
        bc: boundary condition, np.array([dx1, dy1])
        type: "tail": arrow tail at bnd, "head": arrow head at bnd
        scale: scale the length of the arrow
        '''
        if(type == "tail"):
            scale = scale
            bnd_s = bnd + scale*bc
            self.add_arrow("-latex", xy=(bnd, bnd_s))
            self.add_arrow("bar-", xy=(bnd, bnd_s))
        else:
            scale = -scale
            bnd_s = bnd + scale*bc
            self.add_arrow("latex-", xy=(bnd, bnd_s))
            self.add_arrow("-bar", xy=(bnd, bnd_s))
        
        if(text is not None):
            self.add_text((bnd[0] + bnd_s[0])/2, (bnd[1] + bnd_s[1])/2, 
                          text=text, loc=textloc, offset=offset)
        
    def add_dist_bc(self, bnd, bc, type="tail", scale=1, interval=1, 
                    text=None, textloc=None, offset=None):
        '''base function to annotate distributed boundary condition:
        ---------
        | | | | |
        v v v v v
        bnd: boundary nodes, dim: n_nodes*2, np.array([[x1, y1], ..., [xn, yn]])
        bc: boundary conditions, dim: n_nodes*2, np.array([[dx1, dy1], ..., [dxn, dyn]])
        type: "tail": arrow tail at bnd, "head": arrow head at bnd
        scale: scale the length of the arrow
        interval: interval between arrows
        '''
        n_nodes = np.size(bnd, 0)
        if(type == "tail"):
            scale = scale
            arr_type = "-latex"
        else:
            scale = -scale
            arr_type = "latex-"
            
        bnd_s = np.vstack([bnd[:, 0] + scale*bc[:, 0], bnd[:, 1] + scale*bc[:, 1]]).T
            
        for i in range(0, n_nodes, interval):
            self.add_arrow(arr_type, xy=(bnd[i], bnd_s[i]))
        
        self.ax.plot(bnd_s[::interval, 0], bnd_s[::interval, 1], 'k')
        
        if(text is not None):
            self.add_text(bnd_s[n_nodes//2, 0], bnd_s[n_nodes//2, 1], 
                          text=text, loc=textloc, offset=offset)
            
    
    def add_normal_bc():
        #TODO: add_normal_bc based on add_dist_bc, normal to the boundary
        pass