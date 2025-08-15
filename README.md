# CM-VIS: Computational Mechanics Visualization Tools

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](setup.py)

A comprehensive Python package for creating publication-quality visualizations in computational mechanics applications. CM-VIS provides tools for schematic diagram creation, finite element analysis visualization, strength surface plotting, and 3D data processing.

## Features

- **2D/3D Schematic Diagrams**: Create technical drawings with arrows, annotations, and boundary conditions
- **FEM Visualization**: Plot mesh, nodal variables, and elemental variables from simulation results
- **Strength Surface Analysis**: Generate and visualize material failure criteria
- **3D Data Processing**: Handle voxel data and create 3D surface plots
- **Publication Styles**: Built-in matplotlib styles for consistent scientific figures
- **Crack Analysis**: Tools for crack tip trajectory analysis and visualization

## Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/lyyc199586/cm-vis.git
```

### Development Installation
```bash
git clone https://github.com/lyyc199586/cm-vis.git
cd cm-vis
pip install -e .
```

## Requirements

### Core Dependencies
- `numpy >= 1.20`
- `matplotlib >= 3.5`
- `scipy >= 1.7`

### Optional Dependencies
- `netCDF4` - For Exodus file reading (FEM results)
- `scikit-image` - For advanced image processing and strength surfaces
- `s3dlib` - For 3D surface visualization
- `pandas` - For data manipulation

## Gallery

### 2D Schematic Diagrams
Create technical drawings for papers and presentations with professional styling.

#### Discrete crack vs Phase field model

<div style="display: flex; gap: 10px;">
<img src="./png/crack_scheme.png" alt="Crack Scheme" width="200"/>
<img src="./png/pf_scheme.png" alt="Phase Field Scheme" width="200"/>
</div>

#### Brazilian Test Setup
<img src="./png/brz_scheme.png" alt="Brazilian Test Scheme" width="400"/>

### 3D Visualizations

#### 3D Boundary Conditions
<img src="./png/3d_bc.png" alt="3D Boundary Conditions" width="400"/>

#### Voxel image and contourf

<img src="./png/voxel1.png" alt="voxel1" width="400"/>
<img src="./png/voxel2.png" alt="voxel2" width="400"/>

### FEM Post-Processing

#### Brazilian Test Results
<img src="./png/post_brz.png" alt="FEM Results" width="400"/>


### Flowcharts and Diagrams

#### Process Flow Visualization

<img src="./png/flow1.png" alt="Flow Chart 1" width="400"/>
<img src="./png/flow2.png" alt="Flow Chart 2" width="400"/>

*Algorithmic flowcharts and process diagrams*

### Styling Options

#### Styles

<img src="./png/elsevier.png" alt="Elsevier Style" width="400"/>
<img src="./png/sans.png" alt="Sans Style" width="400"/>


#### Custom Axis Formatting
<img src="./png/axis_lines.png" alt="Axis Lines" width="400"/>


