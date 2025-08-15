"""
Finite Element Analysis Visualization
=====================================

This subpackage provides tools for visualizing finite element analysis results
and processing simulation data from various FEM software packages.

Modules
-------
exodus : Exodus file reader for MOOSE-generated results
post : FEM results plotting and visualization
cracktip : Crack tip trajectory analysis tools
nucpfm : Nuclear phase field model utilities

Classes
-------
Exodus : Read and process exodus files
FEMPlotter : Plot variables on FEM meshes
CrackTipAnalyzer : Analyze crack propagation trajectories  
LDL : LDL decomposition utilities for phase field models
"""

from .cracktip import CrackTipAnalyzer
from .nucpfm import LDL
from .post import FEMPlotter
from .exodus import Exodus

__all__ = ['CrackTipAnalyzer', 'Exodus', 'LDL', 'FEMPlotter']