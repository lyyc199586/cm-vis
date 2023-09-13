import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scheme.basic import Scheme

class PFScheme(Scheme):
    
    def add_phasefield(self, crackset, reg_length, cm, domain=None):
        '''add phase field from crackset in domain
        crackset: [crack_path1, crack_path2, ...]
        reg_length: regularization length of phase field
        cm: matplotlib colormap
        domain: path of domain of interest (as a mask)
        '''
        if(domain is not None):
            domain_patch = PathPatch(domain, fc='None', ec='None')
            self.ax.add_patch(domain_patch) # this is necessary for set_clip_path
            
        vertices = np.empty((0, 2))
        codes = []
        for crack_path in crackset:
            vertices = np.vstack((vertices, crack_path.vertices))
            codes.extend(crack_path.codes)
        
        path = Path(vertices, codes)
        
        # iteratively plot crackset of width to form phase field
        num = 20
        colors = cm(np.linspace(0, 1, num))
        for (i, width) in enumerate(np.linspace(reg_length, self.lw, num)):
            patch = PathPatch(path, lw=width, color=colors[i], fc='None',
                              capstyle='round')
            if(domain is not None):
                patch.set_clip_path(domain_patch)
            self.ax.add_patch(patch)
            
        if(domain is not None):
            # re-draw the outer boundary
            self.ax.plot(domain.vertices[:, 0], domain.vertices[:, 1], 'k', lw=self.lw)
        
        return path