""" plot various type of strength sace for PMMA, used in dynamic branching.
see Michael Borden's PhD thesis at p132 and Rudy's paper on coheisve PFM"""

# %% gen
import sys

sys.path.append("../src")
from strength.gen import StrengthSurface
from strength.plot import SurfacePlotter

# material properties of PMMA, unit: MPa, N, mm, s
E = 32e3
nu = 0.2
Gc = 3e-3
rho = 2.54e-9
sigma_ts = 3.08
sigma_cs = 9.24

# surface parameters
vms = {
    "mname": "pmma",
    "stype": "VMS",
    "props": [sigma_ts],
    "srange": [-50, 10, 121],
}

# sace gen and check
ss = [vms]

for s in ss:
    data_dir = f"../data/strength/ss_{s['mname']}_{s['stype']}_props{s['props']}_srange{s['srange']}.npy"
    surface = StrengthSurface(s["stype"], s["props"], s["srange"], data_dir)
    surface.gen()
    plotter = SurfacePlotter(data_dir)
    plotter.plot(dim=2, save=True)


# %% plot
