# template script to read exodus file and plot variable

# %% test for results on traingle mesh
from utils.exodus import Exodus
from utils.post import FEMPlotter

file_dir = "../../temp/damage-1.e"
exodus = Exodus(file_dir)

# read exodus file
t = exodus.get_time()
verts, faces = exodus.get_mesh()
d = exodus.get_var(var_name="d", timestep=30) # test nodal var
psie_active = exodus.get_var(var_name="psie_active", timestep=30) # test elemental var

# plot mesh, nodal variable, and elemental variable
plotter = FEMPlotter(verts, faces)
plotter.plot(lw=0.1)
plotter.plot(d)
plotter.plot(psie_active)

# %% test for results on quad mesh

from utils.exodus import Exodus
from utils.post import FEMPlotter

file_dir = "../../temp/brz_nuc22_p300_a10_l2.5_d5_iref2.e"
exodus = Exodus(file_dir)

# read exodus file
t = exodus.get_time()
verts, faces = exodus.get_mesh()
d = exodus.get_var(var_name="d", timestep=-1) # test nodal var
stress_11 = exodus.get_var(var_name="stress_11", timestep=-1) # test elemental var

# plot mesh, nodal variable, and elemental variable
plotter = FEMPlotter(verts, faces)
plotter.plot(lw=0.1)
plotter.plot(d)
plotter.plot(stress_11)

# %%
