# a wrapper based on netCDF4 to read moose generated exodus files to numpy arrays
# partly inspired by Yan Zhan (2022)

import netCDF4
import numpy as np
import numpy.typing as npt


class Exodus:
    """IO of exodus files, only for 2D for now
    TODO: add support to series of exouds files"""

    def __init__(self, file_dir):
        self.dir = file_dir

    def get_time(self) -> np.ndarray:
        """get time series"""
        model = netCDF4.MFDataset(self.dir)
        time = np.ma.getdata(model.variables["time_whole"][:])

        model.close()
        return time

    def get_block_info(self) -> (int, list):
        """get (block numbers, block names) from exodus file"""
        model = netCDF4.Dataset(self.dir)
        num_blocks = model.dimensions["num_el_blk"].size
        names = []
        for block_id in range(num_blocks):
            name_raw = np.ma.getdata(model.variables["eb_names"][block_id]).astype("U8")
            name = "".join(name_raw)
            names.append(name)

        model.close()
        return (num_blocks, names)

    def get_mesh(self, block_id=0) -> (np.ndarray, np.ndarray):
        """sometimes we only want to read mesh info"""
        model = netCDF4.MFDataset(self.dir)

        # get model info
        dim = model.dimensions["num_dim"].size

        # get coords
        if dim == 2:
            x_coord = np.ma.getdata(model.variables["coordx"][:])
            y_coord = np.ma.getdata(model.variables["coordy"][:])
            verts = np.column_stack([x_coord, y_coord])
        else:
            x_coord = np.ma.getdata(model.variables["coordx"][:])
            y_coord = np.ma.getdata(model.variables["coordy"][:])
            z_coord = np.ma.getdata(model.variables["coordz"][:])
            verts = np.column_stack([x_coord, y_coord, z_coord])

        # get connect matrix
        faces = (
            np.ma.getdata(model.variables[f"connect{block_id + 1}"][:]) - 1
        )  # index start from 0

        model.close()
        return (verts, faces)

    def get_var(self, var_name, timestep=0) -> np.ndarray:
        """get variable with name and timestep"""

        def get_var_names(model, key="name_nod_var"):
            """get varible names:
            for nodel value: key='name_no_var'
            for elemental value: key='name_elem_var'"""
            var_names = []
            for vname in np.ma.getdata(model.variables[key][:]).astype("U8"):
                var_names.append("".join(vname))

            return var_names

        model = netCDF4.MFDataset(self.dir)
        nod_var_names = get_var_names(model, "name_nod_var")
        elem_var_names = get_var_names(model, "name_elem_var")

        # get variable
        if var_name in nod_var_names:
            var_index = nod_var_names.index(var_name)
            vals = model.variables[f"vals_nod_var{var_index+1}"][:][timestep]
        elif var_name in elem_var_names:
            var_index = elem_var_names.index(var_name)
            vals = model.variables[f"vals_elem_var{var_index+1}eb1"][:][timestep]
        else:
            print("No such variable!")
            return None

        return vals
