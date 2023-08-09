""" plot various type of strength sace for PMMA, used in dynamic branching.
see Michael Borden's PhD thesis at p132 and Rudy's paper on coheisve PFM"""

# %% settings
import sys

sys.path.append("../src")

import numpy as np
from strength.gen import StrengthSurface
from strength.plot import SurfacePlotter

# material properties of PMMA, unit: MPa, N, mm, s
E = 32e3
nu = 0.2
Gc = 3e-3
rho = 2.54e-9
sigma_ts = 3.08
sigma_cs = 9.24

K = np.round(E / 3 / (1 - 2 * nu), 2)
mu = np.round(E / 2 / (1 + nu), 2)
lbda = np.round(E * nu / (1 + nu) / (1 - 2 * nu), 2)

ell = 2
delta = 4

# surface parameters
vms = {
    "mname": "pmma",
    "stype": "VMS",
    "props": [sigma_ts],
    "srange": [-50, 10, 121],
}

drucker = {
    "mname": "pmma",
    "stype": "DRUCKER",
    "props": [sigma_ts, sigma_cs],
    "srange": [-50, 10, 121],
}

isotropic = {
    "mname": "pmma",
    "stype": "ISOTROPIC",
    "props": [sigma_ts, mu, K],
    "srange": [-50, 10, 121],
}

voldev = {
    "mname": "pmma",
    "stype": "VOLDEV",
    "props": [sigma_ts, mu, K],
    "srange": [-50, 10, 121],
}

spectral = {
    "mname": "pmma",
    "stype": "SPECTRAL",
    "props": [sigma_ts, lbda, mu, K, nu],
    "srange": [-50, 10, 121],
}

nuc2020 = {
    "mname": "pmma",
    "stype": "KLBFNUC",
    "props": [sigma_ts, sigma_cs, mu, K, Gc, ell, delta],
    "srange": [-50, 10, 121],
}

nuc2022 = {
    "mname": "pmma",
    "stype": "KLRNUC",
    "props": [sigma_ts, sigma_cs, mu, lbda, K, Gc, ell, delta],
    "srange": [-50, 10, 121],
}

# %% surface gen
# ss = [vms, drucker]
ss = [spectral]

for s in ss:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    surface = StrengthSurface(s["stype"], s["props"], s["srange"], data_dir)
    surface.gen()
    plotter = SurfacePlotter(data_dir)
    ax = plotter.plot(dim=3, save=True)

# %% plot settings

ss = [vms, drucker, isotropic, voldev, spectral, nuc2020, nuc2022]
ss1 = [isotropic, voldev, spectral]
ss2 = [drucker, nuc2020, nuc2022]
labels = {
    'VMS':"Von Mises",
    'DRUCKER':"Drucker-Prager",
    'ISOTROPIC':"PFM (isotropic)",
    'VOLDEV':"PFM (vol-dev)",
    'SPECTRAL': "PFM (spectral)",
    'KLBFNUC':"Nuc-PFM (2020)",
    'KLRNUC':"Nuc-PFM (2022)",
}

# %% 2d contour plot
import matplotlib.pyplot as plt

plt.style.use("../misc/elsevier.mplstyle")
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.655))

for s in ss1:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    plotter = SurfacePlotter(data_dir)
    plotter.plot(dim=2, ax=ax[0], label=labels[s["stype"]])
    
for s in ss2:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    plotter = SurfacePlotter(data_dir)
    plotter.plot(dim=2, ax=ax[1], label=labels[s["stype"]])


# annotate
fig.suptitle("Strength surface of PMMA")
for ax_i in ax:
    ax_i.set_xlim([-20, 7.5])
    ax_i.set_ylim([-20, 7.5])
    ax_i.set_aspect("equal")
    ax_i.set_xlabel("$\sigma_{1}$ (MPa)")
    ax_i.set_ylabel("$\sigma_{2}$ (MPa)")
    ax_i.legend(loc='lower left')


#%% 3d isosurface plot
import matplotlib.pyplot as plt

plt.style.use("../misc/elsevier.mplstyle")
fig, ax = plt.subplots(2, 2, figsize=(6.4, 6.4), subplot_kw={"projection": "3d"})
# plt.tight_layout()

ss3 = [[spectral, drucker], [nuc2020, nuc2022]]
colors = [['tab:blue', 'tab:red'], ['tab:orange', 'tab:green']]

for i in range(2):
    for j in range(2):
        s = ss3[i][j]
        data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
        plotter = SurfacePlotter(data_dir)
        plotter.plot(dim=3, ax=ax[i][j], label=labels[s["stype"]], color=colors[i][j], alpha=0.8)
        
        # annotate
        ax[i][j].set_aspect("equal")
        ax[i][j].set_xlabel("$\sigma_{1}$ (MPa)", labelpad=-5)
        ax[i][j].set_ylabel("$\sigma_{2}$ (MPa)", labelpad=-5)
        ax[i][j].set_zlabel("$\sigma_{3}$ (MPa)", labelpad=-5)
        ax[i][j].tick_params(axis='x', pad=-3)
        ax[i][j].tick_params(axis='y', pad=-3)
        ax[i][j].tick_params(axis='z', pad=-1)
        ax[i][j].set_title(labels[s["stype"]])
        

fig.suptitle("Strength surface of PMMA")
# %% save plot
save_dir = "../example/"
fig.savefig(save_dir + "ss_pmma_3d.png")
# %%
