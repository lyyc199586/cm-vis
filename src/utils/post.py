# use matplotlib to do simple plots with FEM mesh and variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection

class FEMPlotter:
    """plot variable on FEM mesh (currently only 2D)"""
    
    def __init__(self, verts, faces) -> None:
        """ verts:[n_nodes, dim], 
            faces:[n_elements, n_nodes_of_element],
            var: [n_nodes, dof] if nodal, [n_elemnts, dof] if elemental"""
        self.verts = verts
        self.faces = faces
        
    def plot(self, var=None, ax=None, **kwargs):
        """return matplotlib axe and plot object"""
        
        n_nodes = np.size(self.verts, 0)
        n_elements = np.size(self.faces, 0)
        n_nodes_of_element = np.size(self.faces, 1)

        def quad_to_tri(quad):
            """convert a quad into 2 traiangles
            """
            tris = [quad[:3], [quad[0], quad[2], quad[3]]]
            return tris
        
        # plot settings
        if(ax is None):
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            
        # only plot mesh if no variable provided
        if(var is None):
            p = PolyCollection([self.verts[face] for face in self.faces], closed=True,
                               antialiaseds=True, facecolor="none", edgecolor="k", **kwargs)
            ax.add_collection(p)
            ax.autoscale()
            return (ax, p)
        else:
            clim = (np.min(var), np.max(var))
        
        # check variable type
        if(np.size(var, 0) == n_nodes):
            var_type = 'nodal'
        elif(np.size(var, 0) == n_elements):
            var_type = 'elemental'
        else:
            print("Unsupported variable type!")
            return None
        
        if(var_type == 'nodal'):
            if(n_nodes_of_element == 3):
                # triangle mesh
                triangles = self.faces
            else:
                tris = []
                for quad in self.faces:
                    triangle = quad_to_tri(quad)
                    tris.extend(triangle)
                triangles = np.array(tris)
                
            # plot nodal value with tripcolor
            tri_obj = tri.Triangulation(self.verts[:, 0], self.verts[:, 1], triangles)
            p = ax.tripcolor(tri_obj, var, clim=clim, cmap=plt.cm.coolwarm, 
                             shading="gouraud", **kwargs)
            
        else:
            # plot elemental variable with PolyCollection
            p = PolyCollection([self.verts[face] for face in self.faces], closed=True, array=var,
                               cmap=plt.cm.coolwarm, antialiaseds=True, edgecolor="face", **kwargs)
            p.set_clim(clim)
            ax.add_collection(p)
            ax.autoscale()
        
        return (ax, p)