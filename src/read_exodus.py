# template script to read exodus file and plot variable

# %%
from utils.exodus import Exodus
from utils.post import FEMPlotter

file_dir = "../../temp/damage-1.e"
exodus = Exodus(file_dir)

# read exodus file
t = exodus.get_time()
verts, faces = exodus.get_mesh()
d = exodus.get_var(var_name="d", timestep=-1)

# plot variable
plotter = FEMPlotter(verts, faces, d)
ax = plotter.plot(dim=2)
# %%
