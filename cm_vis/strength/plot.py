# calculate and plot contour in 2d, isosurface in 3d from

import re
import numpy as np
import matplotlib.pyplot as plt
import s3dlib.surface as s3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, find_contours


class SurfacePlotter:
    """evaluate and plot strength surface"""

    def __init__(self, data_dir: str) -> None:
        """data_dir: path to data.npy"""
        self.dir = data_dir
        self.srange = self.get_srange()
        self.surf = self.get_surf()
        
    def get_srange(self):
            pattern = r"srange\[((?:-?\d+(?:\.\d+)?(?:,\s*)?)+)\]"
            matches = re.findall(pattern, self.dir)
            if matches:
                srange = [float(val) for val in matches[0].split(",")]
            else:
                print("srange not found or wrong format!")
                srange = -1

            return srange
        
    def get_surf(self, norm: float = None, **kwargs):
        """load surf"""

        # evaulate contour of isosurface
        xmin, xmax, num = self.srange
        num = int(num)
        dx = (xmax - xmin) / (num - 1)
        f = np.load(self.dir)

        f_smooth = gaussian_filter(f, sigma=1, order=0)
        verts, faces, _, _ = marching_cubes(f_smooth, level=0)

        # relocate vertices
        verts[:, 0] = (verts[:, 0] * dx + xmin) / (norm if norm else 1)
        verts[:, 1] = (verts[:, 1] * dx + xmin) / (norm if norm else 1)
        verts[:, 2] = (verts[:, 2] * dx + xmin) / (norm if norm else 1)
        
        self.surf = s3d.Surface3DCollection(verts, faces, **kwargs)
        return self.surf

    def plot(self, option: str = "3D", ax=None, save: bool = False, norm: float = None, nu: float = None, s3: float = 0, **kwargs):
        """return ax
        option: "3D", "plane_stress", or "plane_strain"
        save: True to save to csv
        norm: normalize by s1/norm, s2/norm, s3/norm
        nu: Poisson's ratio for plane strain calculation
        s3: value to use for s3 in plane stress condition (default is 0)"""

        # evaulate contour of isosurface
        xmin, xmax, num = self.srange
        num = int(num)
        dx = (xmax - xmin) / (num - 1)
        f = np.load(self.dir)
        
        if option == "3D":

            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_proj_type("ortho")
                if norm is not None:
                    ax.set_xlabel("s1/norm")
                    ax.set_ylabel("s2/norm")
                    ax.set_zlabel("s3/norm")
                else:
                    ax.set_xlabel("s1")
                    ax.set_ylabel("s2")
                    ax.set_zlabel("s3")

            # load surf
            ax.add_collection3d(self.surf.shade())
            ax.set(xlim=[xmin, xmax], ylim=[xmin, xmax], zlim=[xmin, xmax])
            ax.set_title(str(self.surf))

            if save:
                verts_dir = self.dir.replace(".npy", "_3d_verts.csv")
                faces_dir = self.dir.replace(".npy", "_3d_faces.csv")
                np.savetxt(verts_dir, verts * dx + xmin, delimiter=",", fmt="%.2f")
                np.savetxt(faces_dir, faces, delimiter=",", fmt="%d")

        elif option == "plane_strain":
            # Plane strain 2D plot (s1, s2)
            s1_range = np.linspace(xmin, xmax, num)
            s2_range = np.linspace(xmin, xmax, num)
            s1_grid, s2_grid = np.meshgrid(s1_range, s2_range)

            # Calculate s3 = nu * (s1 + s2)
            s3_grid = nu * (s1_grid + s2_grid)
            s3_indices = np.round((s3_grid - xmin) / dx).astype(int)

            # Ensure the indices are within bounds
            s3_indices = np.clip(s3_indices, 0, num - 1)

            # Extract the plane strain surface using advanced indexing
            plane_strain_surface = f[np.arange(num)[:, None], np.arange(num), s3_indices]

            # Apply Gaussian smoothing to the plane strain surface
            plane_strain_surface_smooth = gaussian_filter(plane_strain_surface, sigma=1)

            # Extract the contour of the plane strain surface
            contours = find_contours(plane_strain_surface_smooth, 0)

            if ax is None:
                fig, ax = plt.subplots()
                ax.set_aspect("equal")
                if norm is not None:
                    ax.set_xlabel("s1/norm")
                    ax.set_ylabel("s2/norm")
                else:
                    ax.set_xlabel("s1")
                    ax.set_ylabel("s2")

            for contour in contours:
                ax.plot((contour[:, 0] * dx + xmin) / (norm if norm else 1), (contour[:, 1] * dx + xmin) / (norm if norm else 1), **kwargs)

            if save:
                save_dir_2d = self.dir.replace(".npy", f"2d_plane_strain_nu{nu}.csv")
                np.savetxt(save_dir_2d, np.column_stack((contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin)), delimiter=",", fmt="%.2f")

        elif option == "plane_stress":
            # Plane stress cross-section at s3 = s3 (default is 0)
            z_index = int((s3 - xmin) / dx)  # index at z=0 plane, for plane stress
            contours = find_contours(f[:, :, z_index], 0)

            if ax is None:
                fig, ax = plt.subplots()
                ax.set_aspect("equal")
                if norm is not None:
                    ax.set_xlabel("s11/norm")
                    ax.set_ylabel("s22/norm")
                else:
                    ax.set_xlabel("s11")
                    ax.set_ylabel("s22")

            for contour in contours:
                ax.plot((contour[:, 0] * dx + xmin) / (norm if norm else 1), (contour[:, 1] * dx + xmin) / (norm if norm else 1), **kwargs)

            if save:
                save_dir = self.dir.replace(".npy", f"2d_plane_stress_s3{s3}.csv")
                np.savetxt(save_dir, np.column_stack(contours[0]), delimiter=",", fmt="%.2f")

        return ax
