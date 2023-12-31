#%% plot schematic diagram for Brazilian contact test
from cm_vis.scheme.basic import Scheme

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

fig = plt.figure(figsize=(6.4, 2.655))
plt.style.use("../misc/elsevier.mplstyle")

# plot schematic diagram
with plt.style.context("../misc/fem.mplstyle"):
  ax1 = fig.add_subplot(1, 2, 1)

  disk = Circle([0, 0], 25, fc='gray', ec='k', lw=0.4)
  left_platen = Rectangle([-30, -20], 5, 40, fc='lightgray', ec='k', lw=0.4)
  right_platen = Rectangle([25, -20], 5, 40, fc='lightgray', ec='k', lw=0.4)
  ax1.add_patch(disk)
  ax1.add_patch(left_platen)
  ax1.add_patch(right_platen)
  ax1.set(xlim=[-40, 40], ylim=[-40, 40], aspect='equal', title='Setups')

  # draw scheme
  scheme = Scheme(ax1, lw=0.4)
  scheme.dim_dist([-25, -25], [25, -25], textloc='lower')
  scheme.dim_radius([0, 0], 25, angle=45, text='$r$', textloc='upper')
  scheme.dim_angle(5, 0, 45, center=[0, 0], 
                   text="$\\theta$", arrowloc="stop", textloc="right")
  
  # draw coord axis
  scheme.add_coord_axis(length=np.array([10, 10]))

  # def bc
  left_bnd = np.vstack((-30*np.ones(100), np.linspace(-20, 20, 100))).T
  right_bnd = np.vstack((30*np.ones(100), np.linspace(-20, 20, 100))).T
  bcs = np.vstack((8*np.ones(100), np.zeros(100))).T

  scheme.add_dist_bc(left_bnd, bcs, type="head", scale=1, interval=7, 
                     text="$\overline{u}_x$", textloc="left")
  scheme.add_fix_bc(right_bnd, spacing=4, angle=-45)

# plot boundary condition vs. time
with plt.style.context("../misc/elsevier.mplstyle"):

  t0 = 100 # us
  tf = 200
  u0 = 1 # mm

  t = np.linspace(0, tf, 100)
  load = u0*(-np.cos(np.pi*t/t0) + 1)/2

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(t, load)
  
  ax2.set(title="Load", xlabel="Time, $t (\mu\mathrm{s})$",
          ylabel="Displacement, $\overline{u}_x (\mathrm{mm})$")


# %% save plot
save_dir = "../out/"
fig.savefig(save_dir + "brz_contact_scheme.png")
# %%
