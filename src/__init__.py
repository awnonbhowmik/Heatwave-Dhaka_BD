"""
Climate Analysis Source Package

This package contains modules for comprehensive climate data analysis,
including data loading, statistical analysis, and visualization.

Modules:
    data_loader: Enhanced data loading and preprocessing
    statistical_analysis: Comprehensive statistical testing
    visualization: Publication-quality plotting
    visualization_utils: Specialized visualization tools
    predictive_models: Machine learning models for climate prediction
"""

__version__ = "1.0.0"
__author__ = "Climate Research Team"

from . import (
    data_loader,
    predictive_models,
    statistical_analysis,
    visualization,
    visualization_utils,
)

# Available modules
__all__ = [
    "data_loader",
    "statistical_analysis",
    "visualization",
    "visualization_utils",
    "predictive_models",
]
