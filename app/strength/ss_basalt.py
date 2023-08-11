""" plot various type of strength sace for Bazalt, used in dynamic Brazilian split tests.
For experiments see Yin et al. rock paper (2022). We will use Cohesive-PFM (spectral), 
Cohesive-PFM(voldev), Drucker-Prager (as a baselien for Nuc-PFM), Nuc-PFM(2020), and Nuc-PFM(2022)"""

# %% settings
import sys

sys.path.append("../../src")

import numpy as np
from strength.gen import StrengthSurface
from strength.plot import SurfacePlotter

# material properties of Bazalt, unit: MPa, N, mm, s
E = 20.11e3
nu = 0.24
Gc = 0.1
sigma_ts = 11.31
sigma_cs = 159.08

K = np.round(E / 3 / (1 - 2 * nu), 2)
mu = np.round(E / 2 / (1 + nu), 2)
lbda = np.round(E * nu / (1 + nu) / (1 - 2 * nu), 2)

ell = 1
delta = 0

drucker = {
    "mname": "pmma",
    "stype": "DRUCKER",
    "props": [sigma_ts, sigma_cs],
    "srange": [-300, 30, 331],
}

nuc2022 = {
    "mname": "pmma",
    "stype": "KLRNUC",
    "props": [sigma_ts, sigma_cs, mu, lbda, K, Gc, ell, delta],
    "srange": [-300, 30, 331],
}

# %% surface gen
# ss = [vms, drucker]
ss = [drucker, nuc2022]

for s in ss:
    data_dir = f"../../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    surface = StrengthSurface(s["stype"], s["props"], s["srange"], data_dir)
    surface.gen()
    plotter = SurfacePlotter(data_dir)
    ax = plotter.plot(dim=2, save=True)

# %% 2d contour plot
import matplotlib.pyplot as plt

plt.style.use("../../misc/elsevier.mplstyle")
fig, ax = plt.subplots()

labels = {
    'VMS':"Von Mises",
    'DRUCKER':"Drucker-Prager",
    'ISOTROPIC':"PFM (isotropic)",
    'VOLDEV':"PFM (vol-dev)",
    'SPECTRAL': "PFM (spectral)",
    'KLBFNUC':"Nuc-PFM (2020)",
    'KLRNUC':"Nuc-PFM (2022)",
}

ss1 = [drucker, nuc2022]
for s in ss1:
    data_dir = f"../../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    plotter = SurfacePlotter(data_dir)
    plotter.plot(dim=2, ax=ax, label=labels[s["stype"]])
    
# annotate
fig.suptitle("Strength surface of Basalt")
ax.set_aspect("equal")
ax.set_xlim([-250, 50])
ax.set_ylim([-250, 50])
ax.set_xlabel("$\sigma_{1}$ (MPa)")
ax.set_ylabel("$\sigma_{2}$ (MPa)")
ax.legend(loc='lower left')

# %%
save_dir = "../../out/"
fig.savefig(save_dir + "ss_basalt_2d.png")
# %%
