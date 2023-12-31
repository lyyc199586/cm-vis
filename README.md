# CM-VIS

My computational mechanics visualization scripts

## Installation
```
pip install git+https://github.com/lyyc199586/cm-vis.git
```

## Requirements
* numpy
* scipy
* scikit-image
* matplotlib
* s3dlib
* netCDF4

## Gallery

### FEM: schematic
> plot schematics of simulation setups
> see more in `./examples`
> 
![crack_scheme](out/crack_scheme.png) ![pf_scheme](out/pf_scheme.png)

### FEM: results
> plot mesh, nodal variable and elemental variable on tri and quad mesh from FEM result files.
> see more in `./examples`

![brz_post](./out/post_brz.png)

### Strength surface
> plot strength surface of various type given material properties, including Von Mises, Drucker-Prager, phase field model with no split, V-D split, and spectral split, nucleation phase field models (Kumar et al. [2020](https://doi.org/10.1016/j.jmps.2020.104027) and [2022](https://doi.org/10.1007/s10704-022-00653-z)).
> see `./apps/ss_pmma.py`

For example, plot various type of strength surface of PMMA

* 2D contours
  
![ss_2d_1](./out/ss_pmma_2d.png)

* 3D isosurface
  
![ss_3d_drucker](./out/ss_pmma_3d.png)

The contour line (in 2D) and vertices and faces of isosurface (in 3D) can be output to csv with option `save=True`.
