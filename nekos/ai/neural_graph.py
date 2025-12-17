"""
NeuralGraph - Fast C++ implementation of graph-defined neural networks

This is a thin Python wrapper around the optimized C++ NeuralGraph implementation
with OpenMP parallelization for ~1000x speedup over pure Python.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import nekos


class NeuralGraph:
    """
    A neural network built on graph structure with OpenMP-parallelized propagation.

    This Python class wraps the C++ NeuralGraph implementation for maximum performance.

    Key concepts:
    - Each node is a neuron with activation, bias, and gradient
    - Each edge has a weight representing connection strength
    - Forward pass: parallel propagation through layers
    - Backward pass: parallel gradient computation
    - ~1000x faster than pure Python implementation

    Attributes:
        graph: Underlying nekos.Graph structure
        layer_metadata: Optional layer grouping information
        activations: Current activation values per node (numpy array)
        gradients: Gradient values per node (numpy array)
        biases: Bias term per node (numpy array)
        weights: Weight matrix (numpy array)
    """

    def __init__(
        self,
        graph: 'nekos.Graph',
        layer_metadata: Optional[Dict] = None,
        walk_probability: float = 1.0,
        activation_fn: str = 'tanh',
        learning_rate: float = 0.01,
        num_threads: int = 4,
        seed: Optional[int] = None
    ):
        """
        Initialize NeuralGraph from a nekos.Graph.

        Args:
            graph: Base graph structure
            layer_metadata: Dict with 'layer_order' and 'node_to_layer' for layer structure
            walk_probability: Legacy parameter (not used in C++ version)
            activation_fn: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
            learning_rate: Learning rate for gradient descent
            num_threads: Number of OpenMP threads to use
            seed: Random seed for weight initialization
        """
        self.graph = graph
        self.layer_metadata = layer_metadata
        self.activation_fn_name = activation_fn
        self.walk_probability = walk_probability  # For backward compatibility

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Extract layer structure from metadata
        if layer_metadata is None or 'layer_order' not in layer_metadata:
            raise ValueError("layer_metadata with 'layer_order' is required for C++ backend")

        self.layer_order = layer_metadata['layer_order']

        # Build node_to_layer mapping if not provided
        if 'node_to_layer' in layer_metadata:
            node_to_layer_dict = layer_metadata['node_to_layer']
            # Convert dict to list
            n_nodes = graph.num_nodes()
            node_to_layer_list = [0] * n_nodes
            for node, layer in node_to_layer_dict.items():
                node_to_layer_list[node] = layer
        else:
            # Build from layer_order
            node_to_layer_list = [0] * graph.num_nodes()
            for layer_idx, nodes in enumerate(self.layer_order):
                for node in nodes:
                    node_to_layer_list[node] = layer_idx

        # Map activation function name to C++ enum
        activation_map = {
            'tanh': nekos._core.ActivationFn.TANH,
            'relu': nekos._core.ActivationFn.RELU,
            'sigmoid': nekos._core.ActivationFn.SIGMOID,
            'linear': nekos._core.ActivationFn.LINEAR
        }

        if activation_fn not in activation_map:
            raise ValueError(f"Unknown activation function: {activation_fn}. "
                           f"Must be one of {list(activation_map.keys())}")

        cpp_activation = activation_map[activation_fn]

        # Create C++ NeuralGraph
        self._cpp_ng = nekos._core.NeuralGraph(
            graph,
            self.layer_order,
            node_to_layer_list,
            cpp_activation,
            learning_rate,
            num_threads
        )

    @property
    def activations(self) -> np.ndarray:
        """Get current node activations."""
        return self._cpp_ng.get_activations()

    @property
    def gradients(self) -> np.ndarray:
        """Get current node gradients."""
        return self._cpp_ng.get_gradients()

    @property
    def biases(self) -> np.ndarray:
        """Get node biases."""
        return self._cpp_ng.get_biases()

    @biases.setter
    def biases(self, value: np.ndarray):
        """Set node biases."""
        self._cpp_ng.set_biases(value.astype(np.float32))

    def get_weight_matrix(self) -> np.ndarray:
        """Get weight matrix as (n_nodes, n_nodes) numpy array."""
        return self._cpp_ng.get_weights()

    def set_weight_matrix(self, weights: np.ndarray):
        """Set weight matrix from (n_nodes, n_nodes) numpy array."""
        self._cpp_ng.set_weights(weights.astype(np.float32))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.

        Args:
            inputs: Input activations (1D array matching input layer size)

        Returns:
            outputs: Output activations (1D array matching output layer size)
        """
        inputs_f32 = inputs.astype(np.float32)
        self._cpp_ng.forward(inputs_f32)
        return self._cpp_ng.get_outputs()

    def backward(self, targets: np.ndarray) -> float:
        """
        Backward propagation (compute gradients).

        Args:
            targets: Target outputs (1D array matching output layer size)

        Returns:
            loss: MSE loss value
        """
        targets_f32 = targets.astype(np.float32)
        return self._cpp_ng.backward(targets_f32)

    def update_weights(self, learning_rate: Optional[float] = None):
        """
        Update weights using computed gradients (SGD).

        Args:
            learning_rate: Optional learning rate override
        """
        if learning_rate is not None:
            # Temporarily override learning rate
            old_lr = self._cpp_ng.learning_rate
            self._cpp_ng.learning_rate = learning_rate
            self._cpp_ng.update_weights()
            self._cpp_ng.learning_rate = old_lr
        else:
            self._cpp_ng.update_weights()

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> float:
        """
        Complete training step: forward + backward + update.

        Args:
            inputs: Input data
            targets: Target outputs
            learning_rate: Optional learning rate override

        Returns:
            loss: Training loss
        """
        self.forward(inputs)
        loss = self.backward(targets)
        self.update_weights(learning_rate)
        return loss

    def __repr__(self):
        """String representation."""
        n_nodes = self.graph.num_nodes()
        n_layers = len(self.layer_order)
        n_threads = self._cpp_ng.num_threads
        return (f"NeuralGraph(nodes={n_nodes}, layers={n_layers}, "
                f"activation='{self.activation_fn_name}', threads={n_threads})")
