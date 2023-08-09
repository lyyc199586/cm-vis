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
    ax = plotter.plot(dim=2, save=True)

# %% surface plot
import matplotlib.pyplot as plt

plt.style.use("../misc/elsevier.mplstyle")
fig, ax = plt.subplots()

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

# plot
for s in ss2:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    plotter = SurfacePlotter(data_dir)
    plotter.plot(dim=2, ax=ax, label=labels[s["stype"]])


# annotate
ax.set_xlim([-20, 7.5])
ax.set_ylim([-20, 7.5])
ax.set_aspect("equal")
ax.set_title("Strength surface of PMMA")
ax.set_xlabel("$\sigma_{1}$ (MPa)")
ax.set_ylabel("$\sigma_{2}$ (MPa)")
ax.legend()

# %% save plot
save_dir = "../example/"
ax.figure.savefig(save_dir + "ss_pmma_2d_2.png")
# %%
