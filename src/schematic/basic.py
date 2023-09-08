import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.collections import LineCollection, PatchCollection

# everything here should be scaled with linewidth (lw)

class SchemeBase:
    
    def __init__(self, ax, lw=None) -> None:
        self.ax = ax
        if(lw is None):
            self.lw = 0.4
        else:
            self.lw = lw
    
    def add_arrow(self, type, xyfrom, xyto) -> FancyArrowPatch:
        '''base function to draw arraw
        '''
        
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

        arr = FancyArrowPatch(xyfrom, xyto, arrowstyle=style, fc='k',
                              shrinkA=0, shrinkB=0, lw=self.lw, 
                              joinstyle='miter', capstyle='butt')
        return arr
    
    def add_text(self, textx, texty, text, textfc, loc):
        '''base function to draw text
        '''
        textloc = None
        match loc:
            case "center":
                textloc = dict(ha='center', va='center')
            case "upper":
                textloc = dict(ha='center', va='baseline')
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
                textloc = dict(ha='center', va='baseline')
        self.ax.text(textx, texty, text, textloc, bbox=dict(fc=textfc, ec='none'))
        

class Scheme(SchemeBase):
    
    def dim_dist(self, xyfrom, xyto, text=None, textfc=None, loc=None) -> None:
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
        arr_dim = self.add_arrow("latex-latex", xyfrom, xyto)
        arr_tip = self.add_arrow("bar-bar", xyfrom, xyto)
        self.ax.add_patch(arr_dim)
        self.ax.add_patch(arr_tip)
        # add text
        self.add_text(textx, texty, text, textfc, loc)


    def dim_radius(self, center, radius, angle=45, text=None, textfc=None, loc=None) -> None:
        '''annotate radius for arc:
        --text--|>, if text is None, use str(radius)
        '''
        if(text is None):
            text = str(radius)

        if(textfc is None):
            textfc = 'None'

        xyto = [center[0] + radius*np.cos(angle/180*np.pi), 
                center[1] + radius*np.sin(angle/180*np.pi)]
        arr_dim = self.add_arrow("-latex", center, xyto)
        self.ax.add_patch(arr_dim)
        self.add_text(xyto[0], xyto[1], text, textfc, loc)
        
    def add_fix_bc(self, bnd, scale=1, interval=1, loc=None):
        '''annotate fix boundary with short inclined lines:
        
        ///////
        bnd: boundary nodes, dim: n_nodes*2, [[x1, y1], ..., [xn, yn]]
        loc: "upper", "lower", "left", "right"
        scale: scale the length of the arrow
        interval: interval between arrows
        '''
        shift = None
        match loc:
            case "left":
                shift = -4*scale*self.lw
            case "lower":
                shift = -4*scale*self.lw
            case "right":
                shift = 4*scale*self.lw
            case "upper":
                shift = 4*scale*self.lw
            case _:
                shift = 4*scale*self.lw
            
        bnd_s = np.vstack([bnd[:, 0] + shift, bnd[:, 1] + shift]).T
        segs = [[bnd[i], bnd_s[i]] for i in range(0, np.size(bnd, 0), interval)]
        lines = LineCollection(segs, color='k')
        self.ax.add_collection(lines)
        
    def add_dist_bc(self, bnd, bc, type="tail", scale=1, interval=1, text=None, loc=None):
        '''annotate distributed boundary condition:
        
        ---------
        | | | | |
        v v v v v
        bnd: boundary nodes, dim: n_nodes*2, [[x1, y1], ..., [xn, yn]]
        bc: boundary conditions, dim: n_nodes*2, [[dx1, dy1], ..., [dxn, dyn]]
        type: "tail": arrow tail at bnd, "head": arrow head at bnd
        scale: scale the length of the arrow
        interval: interval between arrows
        '''
        
        arrs = []
        if(type == "tail"):
            scale = scale
            arr_type = "-latex"
        else:
            scale = -scale
            arr_type = "latex-"
            
        bnd_s = np.vstack([bnd[:, 0] + scale*bc[:, 0], bnd[:, 1] + scale*bc[:, 1]]).T
            
        for i in range(0, np.size(bnd, 0), interval):
            arr = self.add_arrow(arr_type, bnd[i], bnd_s[i])
            self.ax.add_patch(arr) # cannot use add_collection, will lost all the features of arrow
        
        self.ax.plot(bnd_s[::interval, 0], bnd_s[::interval, 1], 'k')