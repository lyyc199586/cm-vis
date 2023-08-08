# calculate and plot contour in 2d, isosurface in 3d from

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, find_contours


class SurfacePlotter:
    """evaluate and plot strength surface"""

    def __init__(self, data_dir) -> None:
        self.dir = data_dir

    def plot(self, dim, ax=None, **kwargs):
        def get_srange(data_dir):
            pattern = r"srange\[(-?\d+(?:,\s*-?\d+)*)\]"
            matches = re.findall(pattern, data_dir)
            if matches:
                srange = [int(val) for val in matches[0].split(",")]
                print(f"srange={srange}")
            else:
                print("srange not found or wrong format!")
                srange = -1

            return srange

        # evaulate contour of isosurface
        xmin, xmax, num = get_srange(self.dir)
        dx = (xmax - xmin) / (num - 1)
        z_index = int((num - 1) / 2)  # index at z=0 plane, for 2D plot
        f = np.load(self.dir)

        if dim == 2:
            contours = find_contours(f[:, :, z_index], 0)
        else:
            f_smooth = gaussian_filter(f, sigma=1)
            verts, faces, _, _ = marching_cubes(f_smooth, level=0)

        # plot
        if ax is None:
            if dim == 2:
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

        if dim == 2:
            for contour in contours:
                ax.plot(contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin, **kwargs)
        else:
            ax.plot_trisurf(
                verts[:, 0] * dx + xmin,
                verts[:, 1] * dx + xmin,
                verts[:, 2] * dx + xmin,
                triangles=faces,
                **kwargs,
            )

        return ax
