from setuptools import setup, find_packages

setup(
    name='cm_vis',
    version='1.0.0',
    packages=find_packages(),
    package_data={
      'cm_vis': ['styles/*.mplstyle'],
    },
    install_requires=[
      'numpy',
      'scipy',
      'scikit-image',
      'matplotlib',
      'netCDF4',
    ],
)

