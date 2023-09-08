#%% plot schematic diagram for Brazilian contact test
import sys
sys.path.append("../src")
from schematic.basic import Scheme

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
  scheme.dim_radius([0, 0], 25, angle=55, text='$r$', loc='upper')

  # def bc
  left_bnd = np.vstack((-30*np.ones(100), np.linspace(-20, 20, 100))).T
  right_bnd = np.vstack((30*np.ones(100), np.linspace(-20, 20, 100))).T
  bcs = np.vstack((8*np.ones(100), np.zeros(100))).T

  scheme.add_dist_bc(left_bnd, bcs, type="head", scale=1, interval=7, text="$\overline{u}_x$", loc="left")
  scheme.add_fix_bc(right_bnd, loc="right", interval=4)

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
