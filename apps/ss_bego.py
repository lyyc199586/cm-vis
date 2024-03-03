
# %% 
import numpy as np
from cm_vis.strength.gen import StrengthSurface
from cm_vis.strength.plot import SurfacePlotter

# bego properties
E = 6.26e3
nu = 0.2
Gc = 3.656e-2
sigma_ts = 10
sigma_cs = 37.4 # this is to match cohesive model, it's actully 22 from experiments

K = np.round(E / 3 / (1 - 2 * nu), 2)
mu = np.round(E / 2 / (1 + nu), 2)
lbda = np.round(E * nu / (1 + nu) / (1 - 2 * nu), 2)

ell = 0.5
delta = 10

# for cohesive model
spectral = {
    "mname": "bego",
    "stype": "SPECTRAL",
    "props": [sigma_ts, lbda, mu, K, nu],
    "srange": [-50, 20, 141],
}

# for nuc22
nuc2022 = {
    "mname": "bego",
    "stype": "KLRNUC",
    "props": [sigma_ts, sigma_cs, mu, lbda, K, Gc, ell, delta],
    "srange": [-100, 20, 221],
}

# surface gen
# ss = [spectral, nuc2022]
ss = [nuc2022]

for s in ss:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    surface = StrengthSurface(s["stype"], s["props"], s["srange"])
    surface.gen()
# %%
