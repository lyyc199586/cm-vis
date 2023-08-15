# CM-VIS

My computational mechanics visualization scripts

## Requirements
* numpy
* re
* scipy
* skimage
* matplotlib
* netCDF4

## Gallery

### FEM
> plot mesh, nodal variable and elemental variable on tri and quad mesh from FEM result files.
> see `./app/post_brz.py`

![post_brz](./example/post_brz.png)

### Strength surface
> plot strength surface of various type given material properties, including Von Mises, Drucker-Prager, phase field model with no split, V-D split, and spectral split, nucleation phase field models (Kumar et al. 2020 and 2022).
> see `./app/ss_pmma.py`

For example, plot various type of strength surface of PMMA

* 2D contours
  
![ss_2d_1](./example/ss_pmma_2d.png)

* 3D isosurface
  
![ss_3d_drucker](./example/ss_pmma_3d.png)

The contour line (in 2D) and vertices and faces of isosurface (in 3D) can be output to csv with option `save=True`.
