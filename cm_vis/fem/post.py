# use matplotlib to do simple plots with FEM mesh and variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection


class FEMPlotter:
    """plot variable on FEM mesh (currently only 2D)"""

    def __init__(self, model, ax=None) -> None:
        """model: netCDF dataset
        ax: if None, generate a matplotlib ax
        """
        self.model = model
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.ax.set(aspect="equal")

    def plot(self, var=None, tstep=0, block_id=None, clim=None, cmap=None, **kwargs):
        """block_id: if None, plot over all blocks, return (ax, [p1, p2, ..]),
        otherwise return plot only block at block_id, return (ax, p)
        var: variable to plot, numpy ndarray"
        clim: if None, use [var_max, var_min]
        cmap: if None, use coolwarm"""

        def quad_to_tri(quad):
            """convert a quad into 2 traiangles"""
            tris = [quad[:3], [quad[0], quad[2], quad[3]]]
            return tris

        def plot_block(
            self, block_id=None, tstep=0, var=None, clim=None, cmap=None, **kwargs
        ):
            """this is used to plot at a single block"""
            verts, faces = self.model.get_mesh(block_id=block_id, tstep=tstep)
            n_nodes = np.size(verts, 0)
            n_elements = np.size(faces, 0)
            n_nodes_of_element = np.size(faces, 1)

            # only plot mesh if no variable provided
            if var is None:
                p = PolyCollection(
                    [verts[face] for face in faces],
                    closed=True,
                    antialiaseds=True,
                    facecolor="None",
                    edgecolor="k",
                    **kwargs
                )
                self.ax.add_collection(p)
                return (self.ax, p)
            else:
                if clim is None:
                    clim = (np.min(var), np.max(var))
                if cmap is None:
                    cmap = plt.cm.coolwarm

            # check variable type
            if np.size(var, 0) == n_nodes:
                var_type = "nodal"
            elif np.size(var, 0) == n_elements:
                var_type = "elemental"
            else:
                raise ValueError("Unsupported variable type!")

            if var_type == "nodal":
                if n_nodes_of_element == 3:
                    # triangle mesh
                    triangles = faces
                else:
                    tris = []
                    for quad in faces:
                        triangle = quad_to_tri(quad)
                        tris.extend(triangle)
                    triangles = np.array(tris)

                # plot nodal value with tripcolor
                tri_obj = tri.Triangulation(verts[:, 0], verts[:, 1], triangles)
                p = self.ax.tripcolor(
                    tri_obj, var, clim=clim, cmap=cmap, shading="gouraud", **kwargs
                )

            else:
                # plot elemental variable with PolyCollection
                p = PolyCollection(
                    [verts[face] for face in faces],
                    closed=True,
                    array=var,
                    cmap=cmap,
                    antialiaseds=True,
                    edgecolor="face",
                    **kwargs
                )
                p.set_clim(clim)
                self.ax.add_collection(p)

            return (self.ax, p)

        if block_id is None:
            p_lists = []
            block_nums, _ = self.model.get_block_info()
            if block_nums == 1:
                _, p = plot_block(self, 0, tstep, var, clim, cmap, **kwargs)
                self.ax.autoscale(enable=True)
                return (self.ax, p)
            else:
                for i in range(block_nums):
                    _, p = plot_block(self, i, tstep, var, clim, cmap, **kwargs)
                    p_lists.append(p)
                self.ax.autoscale(enable=True)
                return (self.ax, p_lists)
        else:
            _, p = plot_block(self, block_id, tstep, var, clim, cmap, **kwargs)
            self.ax.autoscale(enable=True)
            return (self.ax, p)
