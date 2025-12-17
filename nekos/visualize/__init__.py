"""
Visualization utilities for graphs and neural networks

This module provides visualization functions for:
- Graph visualization (OpenGL-accelerated via vispy)
- Neural network visualization (activations, gradients, weights)
"""

# Graph visualization
from .graph import visualize, hex_to_rgba, generate_cluster_colors

# Neural network visualization
from .neural import (
    visualize_activations,
    visualize_gradients,
    visualize_weights,
    visualize_learning,
    compare_architectures,
    architecture_stats,
    print_architecture_stats
)

__all__ = [
    # Graph visualization
    'visualize',
    'hex_to_rgba',
    'generate_cluster_colors',

    # Neural network visualization
    'visualize_activations',
    'visualize_gradients',
    'visualize_weights',
    'visualize_learning',
    'compare_architectures',
    'architecture_stats',
    'print_architecture_stats',
]
