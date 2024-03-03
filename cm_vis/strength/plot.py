# calculate and plot contour in 2d, isosurface in 3d from

import re
import numpy as np
import matplotlib.pyplot as plt
import s3dlib.surface as s3d
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, find_contours


class SurfacePlotter:
    """evaluate and plot strength surface"""

    def __init__(self, data_dir: str) -> None:
        """data_dir: path to data.npy"""
        self.dir = data_dir

    def plot(self, dim: int, ax=None, save: bool=False, norm: float=None, **kwargs):
        """return ax
        dim: 2 or 3
        save: True to save to csv
        norm: normalize by s1/norm, s2/norm, s3/norm"""
        def get_srange(data_dir):
            # pattern = r"srange\[(-?\d+(?:,\s*-?\d+)*)\]"
            pattern = r"srange\[((?:-?\d+(?:\.\d+)?(?:,\s*)?)+)\]"
            matches = re.findall(pattern, data_dir)
            if matches:
                srange = [float(val) for val in matches[0].split(",")]
            else:
                print("srange not found or wrong format!")
                srange = -1

            return srange

        # evaulate contour of isosurface
        xmin, xmax, num = get_srange(self.dir)
        dx = (xmax - xmin) / (num - 1)
        z_index = int((0 - xmin) / dx)  # index at z=0 plane, for 2D plot
        f = np.load(self.dir)

        if(dim == 2):
            contours = find_contours(f[:, :, z_index], 0)
            if(save):
                save_dir = self.dir.replace(".npy", "_2d.csv")
                x_coord = contours[0][:, 0] * dx + xmin
                y_coord = contours[0][:, 1] * dx + xmin
                coords = np.column_stack((x_coord, y_coord))
                np.savetxt(save_dir, coords, delimiter=",", comments="", fmt="%.2f")

        else:
            f_smooth = gaussian_filter(f, sigma=1, order=0)
            verts, faces, _, _ = marching_cubes(f_smooth, level=0)
            if(save):
                verts_dir = self.dir.replace(".npy", "_3d_verts.csv")
                faces_dir = self.dir.replace(".npy", "_3d_faces.csv")
                x_coord = verts[:, 0] * dx + xmin
                y_coord = verts[:, 1] * dx + xmin
                z_coord = verts[:, 2] * dx + xmin
                coords = np.column_stack((x_coord, y_coord, z_coord))
                np.savetxt(verts_dir, coords, delimiter=",", comments="", fmt="%.2f")
                np.savetxt(faces_dir, faces, delimiter=",", comments="", fmt="%d")

        # plot
        if(ax is None):
            if(dim == 2):
                fig, ax = plt.subplots()
                ax.set_aspect("equal")
                ax.set_xlabel("s11")
                ax.set_ylabel("s22")
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_xlabel("s11")
                ax.set_ylabel("s22")
                ax.set_zlabel("s33")
                ax.set_proj_type("ortho")

        if dim == 2:
            for contour in contours:
                if(norm is not None):
                    ax.plot((contour[:, 0] * dx + xmin)/norm, (contour[:, 1] * dx + xmin)/norm, **kwargs)
                else:
                    ax.plot(contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin, **kwargs)
        else:
            surface = s3d.Surface3DCollection(verts, faces, **kwargs)
            ax.add_collection3d(surface)

        return ax
