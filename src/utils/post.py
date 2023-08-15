# use matplotlib to do simple plots with FEM mesh and variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class FEMPlotter:
    """plot variable on FEM mesh (currently only 2D)"""
    
    def __init__(self, verts, faces, var) -> None:
        """ verts:[N, dim], 
            faces:[N, n_nodes_of_element],
            var: [N, dim]"""
        self.verts = verts
        self.faces = faces
        self.var = var
        
    def plot(self, dim:int, ax=None, **kwargs):
        """return matplotlib axe"""
        
        n_nodes_of_element = np.size(self.faces, 1)
        
        if(ax is None):
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        
        if(n_nodes_of_element == 3):
            # triangle mesh
            tri_obj = tri.Triangulation(self.verts[:, 0], self.verts[:, 1], self.faces)
            ax.tripcolor(tri_obj, self.var, cmap=plt.cm.jet, **kwargs)
            
        else:
            # quadrilateral mesh
            pass 
        
        return ax