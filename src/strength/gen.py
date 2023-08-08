# generate strength surface data of various types

import numpy as np


class StrengthSurface:
    """generate strength surface data of certain types"""

    def __init__(self, type, props, srange, data_dir) -> None:
        """type: strength surface type
        props: material properties
        srange: [xmin, xmax, num]
        data_dir: path to store data.npy"""

        self.type = type
        self.props = props
        self.range = srange
        self.dir = data_dir

    def gen(self):
        def vms(s1, s2, s3, sigma_y):
            """calculate von mises yield surface
            props : [sigma_y]"""
            sigma_v = np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
            f = sigma_v - sigma_y

            return f

        # generate data space
        x_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        y_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        z_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        s1, s2, s3 = np.meshgrid(x_, y_, z_, indexing="xy")

        # generate strength surface data
        match self.type:
            case "VMS":
                sigma_y = self.props[0]
                f = vms(s1, s2, s3, sigma_y)

        # save data.npy
        np.save(self.dir, f)
