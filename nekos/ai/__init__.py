"""
nekos.ai - Neural network on graph structures with random walk propagation

This module treats graphs as neural computation substrates where:
- Nodes are neurons with activation states
- Edges are weighted connections with transition probabilities
- Forward propagation is a random walk through the graph
- Standard MLP is a DAG with walk probability = 1.0
"""

from .neural_graph import NeuralGraph
from .layers import LayerBuilder
from .pytorch_bridge import to_pytorch, from_pytorch, train_with_pytorch
from .visualize import (
    visualize_activations,
    visualize_gradients,
    visualize_weights,
    visualize_learning,
    compare_architectures,
    architecture_stats,
    print_architecture_stats
)

__all__ = [
    'NeuralGraph',
    'LayerBuilder',
    'to_pytorch',
    'from_pytorch',
    'train_with_pytorch',
    'visualize_activations',
    'visualize_gradients',
    'visualize_weights',
    'visualize_learning',
    'compare_architectures',
    'architecture_stats',
    'print_architecture_stats',
]
