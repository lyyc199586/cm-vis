import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Arc, FancyArrowPatch, ArrowStyle, PathPatch
from basic import SchemeBase

class Arrow3D(FancyArrowPatch):
    # from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self.xyz = [[x0, y0, z0], [x1, y1, z1]]
        
    def draw(self, renderer):
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        (x0, y0, z0), (x1, y1, z1) = self.xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)
        

class Scheme3DBase():
    
    def __init__(self, ax, lw=0.4):
        self.ax = ax
        self.lw = lw
        
    def add_arrow3d(self, type, xyz, fc=None):
        '''base function to draw arrow in 3D:
        type: customized matplotlib arrow type,
        xyz=[[x0, y0, z0], [x1, y1, z1]]
            use posA=xyz[0], posB=xyz[1] for straight path, only support straing path in 3D!
        fc: facecolor of arrow
        '''
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
    
    def add_text3d(self, xyz, text, zdir=None, textc=None, boxfc=None, offset=None):
        '''base function to draw text in 3D, a wrapper of ax.text
        offset=[dx, dy, dz]
        '''
        #TODO: try to add support for offset like in 2D
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
    
    def add_coord_axis3d(self, origin=[0, 0, 0], length=[1.0, 1.0, 1.0], 
                       text=['$x$', '$y$', '$z$'], textc=None, shift=1):
        '''draw coordinates at origin (or any location)
        shift: factor to time with the length for text
        '''
        for i in range(np.size(length)):
            xyto = origin.copy()
            xyto[i] = xyto[i] + length[i]
            self.add_arrow3d("-latex", xyz=(origin, xyto))
            self.add_text3d(np.array(xyto)*shift, text[i], textc=textc)

class Scheme3D(Scheme3DBase):
    pass