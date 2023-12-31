""" FEM postprocessing of dynamic Brazilian split tests"""

# %% settings
from cm_vis.fem.exodus import Exodus
from cm_vis.fem.post import FEMPlotter

file_dir = "../../temp/brz_nuc22_p300_a10_l2.5_d5_iref2.e"
exodus = Exodus(file_dir)

# read exodus file
t = exodus.get_time()
d = exodus.get_var(var_name="d", timestep=-1)
stress_11 = exodus.get_var(var_name="stress_11", timestep=-1)

#%% plot
import matplotlib.pyplot as plt

plt.style.use("../misc/fem.mplstyle")
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.655))

plot_vars = [None, d, stress_11]

for i, ax in enumerate(axes):
    plotter = FEMPlotter(exodus, ax=ax)
    _, p = plotter.plot(plot_vars[i], lw=0.1)
    ax.set(aspect="equal", xticklabels=[], yticklabels=[])
    if(plot_vars[i] is not None):
        ax.figure.colorbar(p, ax=ax, fraction=0.046, pad=0.1, orientation='horizontal')

axes[0].set_title("Mesh")
axes[1].set_title("Damage")
axes[2].set_title("Stress yy")
fig.suptitle("FEM visualization of Brazilian split test")

# %%
# %% save plot
save_dir = "../out/"
fig.savefig(save_dir + "post_brz.png")
# %%
