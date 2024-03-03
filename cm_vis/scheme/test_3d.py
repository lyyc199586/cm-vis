#%%
from basic3d import Scheme3D
import matplotlib.pyplot as plt

plt.style.use("../../misc/fem.mplstyle")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,1)
ax.axis('off')


scheme = Scheme3D(ax, lw=0.4)
scheme.add_arrow3d("latex-latex", xyz=[[0, 0, 0], [-1, 1, 1]])
scheme.add_arrow3d("bar-bar", xyz=[[-1, 1, 1], [1, 1, 1]])
scheme.add_text3d([0, 0, 0.5], "(0, 0, 0.5)", offset=[0.1, 0.1, 0.1])
scheme.add_coord_axis3d(shift=1.1)
# ax

# %%
