# template script to plot 2d and 3d strength surface from data

# %%
from strength.plot import SurfacePlotter

file_name = "../data/strength/ss_pmma_VMS_props[10]_srange[-100, 100, 201]"
data_dir = file_name + ".npy"

plotter = SurfacePlotter(data_dir)
ax = plotter.plot(dim=3)

# %%
