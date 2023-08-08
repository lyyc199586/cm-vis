# generate strength surface of various type

# %%
from strength.gen import StrengthSurface

mname = 'pmma'
stype = 'VMS'
props = [10]
srange = [-100, 100, 201]
data_dir = f"../data/strength/ss_{mname}_{stype}_props{props}_srange{srange}.npy"

surface = StrengthSurface(stype, props, srange, data_dir)
surface.gen()

# %%
