import numpy as np
import matplotlib
import matplotlib.patheffects as patheffects
from matplotlib.path import Path
from matplotlib.patches import Arc, FancyArrowPatch, ArrowStyle, ConnectionStyle
from matplotlib.collections import LineCollection, PatchCollection

# everything here should be scaled with linewidth (lw)

class SchemeBase:
    
    def __init__(self, ax, lw=None) -> None:
        self.ax = ax
        if(lw is None):
            self.lw = 0.4
        else:
            self.lw = lw
    
    def add_arrow(self, type, xyfrom, xyto):
        '''base function to draw straight arraw
        type: customized matplotlib arrow type
        '''
        arrow_props = dict(posA=xyfrom, posB=xyto, shrinkA=0, shrinkB=0, lw=self.lw,
                           joinstyle="miter", capstyle="butt", fc='k')
        match type:
            case "-latex":
                style = ArrowStyle('-|>', head_length=6*self.lw, head_width=2*self.lw)
            case "latex-":
                style = ArrowStyle('<|-', head_length=6*self.lw, head_width=2*self.lw)
            case "latex-latex":
                style = ArrowStyle('<|-|>', head_length=6*self.lw, head_width=2*self.lw)
            case "bar-bar":
                style = ArrowStyle('|-|', widthA=4*self.lw, widthB=4*self.lw)
            case _:
                style = ArrowStyle('-')
        
        # assign props and draw arrow
        arrow_props.update(arrowstyle=style)
        arr = FancyArrowPatch(**arrow_props)
        self.ax.add_patch(arr)
    
    def add_text(self, textx, texty, text, textfc=None, loc=None):
        '''base function to draw text, a wrapper of ax.text
        '''
        textloc = None
        if(textfc is None):
            textfc = 'None'
            
        match loc:
            case "center":
                textloc = dict(ha='center', va='center')
            case "upper":
                textloc = dict(ha='center', va='bottom')
                texty = texty + 2*self.lw
            case "lower":
                textloc = dict(ha='center', va='top')
                texty = texty - 2*self.lw
            case "left":
                textloc = dict(ha='right', va='center')
                textx = textx - 2*self.lw
            case "right":
                textloc = dict(ha='left', va='center')
                textx = textx + 2*self.lw
            case _:
                textloc = dict(ha='center', va='bottom')
        self.ax.text(textx, texty, text, textloc, bbox=dict(fc=textfc, ec='none'))
        
    def add_coord_axis(self, origin=np.array([0, 0]), length=np.array([1, 1]), 
                       text=['$x$', '$y$']):
        '''draw coordinates at origin
        '''
        textloc = ["right", "upper"]
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = xyto[i] + length[i]
            self.add_arrow("-latex", origin, xyto)
            self.add_text(xyto[0], xyto[1], text[i], loc=textloc[i])
            
    def add_curve(self, verts, **kwargs):
        '''draw bezier curves passing vertices
        '''
        pass
    
    def add_polygon():
        pass
    
class Scheme(SchemeBase):
    
    def dim_dist(self, xyfrom, xyto, text=None, textfc=None, textloc=None):
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

        textx = (xyfrom[0] + xyto[0])/2
        texty = (xyfrom[1] + xyto[1])/2

        # add arrows
        self.add_arrow("latex-latex", xyfrom, xyto)
        self.add_arrow("bar-bar", xyfrom, xyto)
        # add text
        self.add_text(textx, texty, text, textfc, loc=textloc)

    def dim_radius(self, center, radius, angle=45, text=None, textfc=None, textloc=None):
        '''annotate radius for arc:
        --text--|>, if text is None, use str(radius)
        '''
        if(text is None):
            text = str(radius)

        xyto = [center[0] + radius*np.cos(angle/180*np.pi), 
                center[1] + radius*np.sin(angle/180*np.pi)]
        self.add_arrow("-latex", center, xyto)
        self.add_text(xyto[0]/2, xyto[1]/2, text, textfc, loc=textloc)
        
    def dim_angle(self, radius, start_deg, stop_deg, xyfrom=None, center=None, 
                  arrowloc="stop", text=None, textloc=None):
        '''annotate angle for arc: just like \draw(x, y) arc (start_deg:stop_deg:radius)
        |<--text-->|, if text is None, use str(abs(stop_deg - start_deg))
        provide either xyfrom=[x0, y0] or center=[x0, y0]
        arrowloc: "stop", "start", "both" or "None"
        '''
        if((xyfrom is None and center is None) or (xyfrom is None and center is None)):
            raise ValueError("Provide either xyfrom=[x0, y0] or center=[x0, y0]")
        
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
        match arrowloc:
            case "stop":
                style = ArrowStyle('-|>', head_length=6*self.lw, head_width=2*self.lw)
            case "start":
                style = ArrowStyle('<|-', head_length=6*self.lw, head_width=2*self.lw)
            case "both":
                style = ArrowStyle('<|-|>', head_length=6*self.lw, head_width=2*self.lw)
            case _:
                style = ArrowStyle('-')
        
        arrow_props = dict(path=arc_path, shrinkA=0, shrinkB=0, lw=self.lw,
                           arrowstyle=style, joinstyle="miter", capstyle="butt", 
                           fc='k')
        arr = FancyArrowPatch(**arrow_props)
        self.ax.add_patch(arr)
        
        # add text
        textxy = [center[0] + radius*np.cos((start + stop)/2), 
                  center[1] + radius*np.sin((start + stop)/2)]
        self.add_text(textxy[0], textxy[1], text, loc=textloc)
    
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
        
    def add_dist_bc(self, bnd, bc, type="tail", scale=1, interval=1, text=None, textloc=None):
        '''base function to annotate distributed boundary condition:
        ---------
        | | | | |
        v v v v v
        bnd: boundary nodes, dim: n_nodes*2, [[x1, y1], ..., [xn, yn]]
        bc: boundary conditions, dim: n_nodes*2, [[dx1, dy1], ..., [dxn, dyn]]
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
            self.add_arrow(arr_type, bnd[i], bnd_s[i])
        
        self.ax.plot(bnd_s[::interval, 0], bnd_s[::interval, 1], 'k')
        
        if(text is not None):
            self.add_text(bnd_s[n_nodes//2, 0], bnd_s[n_nodes//2, 1], text=text, loc=textloc)
            
    
    def add_normal_bc():
        #TODO: add_normal_bc based on add_dist_bc, normal to the boundary
        pass