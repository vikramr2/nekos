"""
Core NeuralGraph class - treats graph as a neural network with random walk propagation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import nekos


class NeuralGraph:
    """
    A neural network built on graph structure with random walk propagation.

    Key concepts:
    - Each node is a neuron with activation, bias, and gradient
    - Each edge has a weight and represents a transition probability
    - Forward pass: random walk from input nodes to output nodes
    - Walk probability = 1.0 gives deterministic feedforward (standard MLP)
    - Walk probability < 1.0 gives stochastic propagation

    Attributes:
        graph: Underlying nekos.Graph structure
        clustering: Optional layer grouping (cluster = layer)
        activations: Current activation values per node
        gradients: Gradient values per node (for backprop)
        biases: Bias term per node
        weights: Edge weights as dict (src, dst) -> weight
        walk_probability: Probability of following each edge [0, 1]
    """

    def __init__(
        self,
        graph: 'nekos.Graph',
        layer_metadata: Optional[Dict] = None,
        walk_probability: float = 1.0,
        activation_fn: str = 'tanh',
        seed: Optional[int] = None
    ):
        """
        Initialize NeuralGraph from a nekos.Graph.

        Args:
            graph: Base graph structure
            layer_metadata: Optional dict with 'layer_order' (list of node lists per layer)
            walk_probability: Probability [0,1] of following edges (1.0 = deterministic)
            activation_fn: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
            seed: Random seed for stochastic walks
        """
        self.graph = graph
        self.layer_metadata = layer_metadata
        self.walk_probability = walk_probability
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Initialize node states
        n_nodes = graph.num_nodes()
        self.activations = np.zeros(n_nodes, dtype=np.float32)
        self.gradients = np.zeros(n_nodes, dtype=np.float32)
        self.biases = np.random.randn(n_nodes).astype(np.float32) * 0.01

        # Initialize edge weights
        # Store as dict for now (can optimize to parallel arrays later)
        self.weights = {}
        self._initialize_weights()

        # Set activation function
        self.activation_fn_name = activation_fn
        self.activation_fn, self.activation_derivative = self._get_activation_fn(activation_fn)

        # Track layer order if metadata provided
        self.layer_order = None
        if layer_metadata is not None and 'layer_order' in layer_metadata:
            self.layer_order = layer_metadata['layer_order']

    def _initialize_weights(self):
        """Initialize edge weights with Xavier initialization."""
        for node in range(self.graph.num_nodes()):
            neighbors = self.graph.neighbors(node)
            if len(neighbors) > 0:
                # Xavier initialization: scale by sqrt(fan_in)
                scale = np.sqrt(1.0 / len(neighbors))
                for neighbor in neighbors:
                    # Store weight for directed edge node -> neighbor
                    self.weights[(node, neighbor)] = np.random.randn() * scale

    def _get_activation_fn(self, name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        if name == 'tanh':
            return (
                lambda x: np.tanh(x),
                lambda x: 1 - np.tanh(x)**2
            )
        elif name == 'relu':
            return (
                lambda x: np.maximum(0, x),
                lambda x: (x > 0).astype(np.float32)
            )
        elif name == 'sigmoid':
            sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return (
                sig,
                lambda x: sig(x) * (1 - sig(x))
            )
        elif name == 'linear':
            return (
                lambda x: x,
                lambda x: np.ones_like(x)
            )
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(
        self,
        inputs: np.ndarray,
        input_nodes: Optional[List[int]] = None,
        stochastic: bool = None
    ) -> np.ndarray:
        """
        Forward propagation through the graph.

        Args:
            inputs: Input values (if input_nodes specified) or full activation vector
            input_nodes: Which nodes receive input (None = use first layer from clustering)
            stochastic: Whether to use stochastic walk (None = use self.walk_probability)

        Returns:
            activations: Activation values for all nodes
        """
        if stochastic is None:
            stochastic = self.walk_probability < 1.0

        # Determine input nodes
        if input_nodes is None:
            if self.layer_order is not None:
                input_nodes = self.layer_order[0]
            else:
                raise ValueError("Must specify input_nodes or provide clustering")

        # Set input activations
        if len(inputs) == self.graph.num_nodes():
            # Full activation vector provided
            self.activations = inputs.copy()
        else:
            # Set specific input nodes
            self.activations.fill(0)
            self.activations[input_nodes] = inputs

        # Propagate through layers or graph
        if self.layer_order is not None:
            # Layer-wise propagation (more efficient for DAG-like structures)
            self._forward_layered(stochastic)
        else:
            # General graph propagation
            self._forward_general(stochastic)

        return self.activations.copy()

    def _forward_layered(self, stochastic: bool):
        """Forward pass for layered/clustered graphs."""
        for layer_idx in range(1, len(self.layer_order)):
            current_layer = self.layer_order[layer_idx]

            for node in current_layer:
                # Get incoming neighbors (from previous layers)
                neighbors = self.graph.neighbors(node)

                # Compute weighted sum
                z = self.biases[node]
                for neighbor in neighbors:
                    # Check if we should follow this edge (stochastic walk)
                    if stochastic and np.random.rand() > self.walk_probability:
                        continue

                    weight = self.weights.get((neighbor, node), 0.0)
                    z += weight * self.activations[neighbor]

                # Apply activation function
                self.activations[node] = self.activation_fn(z)

    def _forward_general(self, stochastic: bool, max_iterations: int = 10):
        """
        Forward pass for general graphs without layer structure.
        Uses iterative propagation until convergence.
        """
        prev_activations = self.activations.copy()

        for _ in range(max_iterations):
            new_activations = self.activations.copy()

            for node in range(self.graph.num_nodes()):
                neighbors = self.graph.neighbors(node)

                z = self.biases[node]
                for neighbor in neighbors:
                    if stochastic and np.random.rand() > self.walk_probability:
                        continue

                    weight = self.weights.get((neighbor, node), 0.0)
                    z += weight * prev_activations[neighbor]

                new_activations[node] = self.activation_fn(z)

            # Check convergence
            if np.allclose(new_activations, prev_activations, rtol=1e-4):
                break

            prev_activations = new_activations
            self.activations = new_activations

    def backward(
        self,
        targets: np.ndarray,
        output_nodes: Optional[List[int]] = None,
        loss_fn: str = 'mse'
    ) -> float:
        """
        Backward propagation to compute gradients.

        Args:
            targets: Target values for output nodes
            output_nodes: Which nodes are outputs (None = use last layer)
            loss_fn: Loss function ('mse', 'cross_entropy')

        Returns:
            loss: Computed loss value
        """
        # Determine output nodes
        if output_nodes is None:
            if self.layer_order is not None:
                output_nodes = self.layer_order[-1]
            else:
                raise ValueError("Must specify output_nodes or provide clustering")

        # Compute output gradients and loss
        self.gradients.fill(0)

        if loss_fn == 'mse':
            # Mean squared error
            output_activations = self.activations[output_nodes]
            loss = np.mean((output_activations - targets)**2)
            # Gradient: d(MSE)/d(output) = 2 * (output - target) / n
            self.gradients[output_nodes] = 2 * (output_activations - targets) / len(targets)
        elif loss_fn == 'cross_entropy':
            # Cross-entropy (assumes softmax output)
            output_activations = self.activations[output_nodes]
            # Numerical stability
            output_activations = np.clip(output_activations, 1e-7, 1 - 1e-7)
            loss = -np.mean(targets * np.log(output_activations))
            # For softmax + cross-entropy: gradient = output - target
            self.gradients[output_nodes] = output_activations - targets
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Backpropagate through layers
        if self.layer_order is not None:
            self._backward_layered()
        else:
            self._backward_general()

        return loss

    def _backward_layered(self):
        """Backward pass for layered graphs."""
        # Iterate backward through layers
        for layer_idx in range(len(self.layer_order) - 1, 0, -1):
            current_layer = self.layer_order[layer_idx]

            for node in current_layer:
                # Apply activation derivative
                grad = self.gradients[node] * self.activation_derivative(self.activations[node])

                # Propagate to previous layer
                neighbors = self.graph.neighbors(node)
                for neighbor in neighbors:
                    weight = self.weights.get((neighbor, node), 0.0)
                    # Accumulate gradient for previous layer
                    self.gradients[neighbor] += grad * weight

    def _backward_general(self):
        """Backward pass for general graphs."""
        # For general graphs, we need to handle cycles carefully
        # Simple approach: apply activation derivative to all gradients
        for node in range(self.graph.num_nodes()):
            if self.gradients[node] != 0:
                self.gradients[node] *= self.activation_derivative(self.activations[node])

    def update_weights(self, learning_rate: float = 0.01):
        """
        Update weights using computed gradients (simple SGD).

        Args:
            learning_rate: Step size for gradient descent
        """
        for (src, dst), weight in self.weights.items():
            # Gradient for weight = gradient at dst * activation at src
            grad = self.gradients[dst] * self.activations[src]
            self.weights[(src, dst)] = weight - learning_rate * grad

        # Update biases
        self.biases -= learning_rate * self.gradients

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        input_nodes: Optional[List[int]] = None,
        output_nodes: Optional[List[int]] = None,
        loss_fn: str = 'mse'
    ) -> float:
        """
        Single training step: forward, backward, update.

        Returns:
            loss: Loss value for this step
        """
        # Forward pass
        self.forward(inputs, input_nodes)

        # Backward pass
        loss = self.backward(targets, output_nodes, loss_fn)

        # Update weights
        self.update_weights(learning_rate)

        return loss

    def get_weight_matrix(self) -> np.ndarray:
        """
        Export weights as dense adjacency matrix.
        Useful for visualization and PyTorch conversion.
        """
        n = self.graph.num_nodes()
        W = np.zeros((n, n), dtype=np.float32)
        for (src, dst), weight in self.weights.items():
            W[src, dst] = weight
        return W

    def set_weight_matrix(self, W: np.ndarray):
        """Import weights from dense adjacency matrix."""
        for src in range(self.graph.num_nodes()):
            neighbors = self.graph.neighbors(src)
            for dst in neighbors:
                if W[src, dst] != 0:
                    self.weights[(src, dst)] = W[src, dst]

    def __repr__(self):
        n_nodes = self.graph.num_nodes()
        n_edges = len(self.weights)
        layers = len(self.layer_order) if self.layer_order else "unknown"
        return (
            f"NeuralGraph(nodes={n_nodes}, edges={n_edges}, layers={layers}, "
            f"walk_prob={self.walk_probability}, activation={self.activation_fn_name})"
        )
