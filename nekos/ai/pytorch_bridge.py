"""
PyTorch bridge for training graph-defined neural network architectures

This is NOT a Graph Neural Network (GNN). Instead, this uses graphs to define
the ARCHITECTURE of a neural network - exotic connectivity patterns, sparse networks,
skip connections, etc. The graph structure defines which neurons connect to which.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


if TORCH_AVAILABLE:
    class GraphDefinedNetwork(nn.Module):
        """
        PyTorch module for networks with graph-defined connectivity.

        The graph structure defines the network architecture:
        - Nodes = neurons
        - Edges = weighted connections
        - Clustering = layer grouping

        This enables training exotic topologies:
        - Sparse random networks
        - Small-world networks
        - Networks with skip connections
        - Irregular connectivity patterns

        Then visualize with nekos' force-directed layout!
        """

        def __init__(
            self,
            neural_graph: 'NeuralGraph',
            device: Optional[str] = None
        ):
            """
            Create PyTorch module from NeuralGraph.

            Args:
                neural_graph: Source neural graph (defines architecture)
                device: 'cuda', 'cpu', or None (auto-detect)
            """
            super().__init__()

            self.neural_graph = neural_graph
            self.n_nodes = neural_graph.graph.num_nodes()

            # Auto-detect device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)

            # Create weight matrix as parameter
            # Only connections that exist in graph have non-zero weights
            W = neural_graph.get_weight_matrix()
            self.weight_matrix = nn.Parameter(
                torch.from_numpy(W).to(self.device)
            )

            # Create bias vector as parameter
            self.bias = nn.Parameter(
                torch.from_numpy(neural_graph.biases).to(self.device)
            )

            # Store activation function
            self.activation_name = neural_graph.activation_fn_name

            # Store layer structure
            self.layer_order = neural_graph.layer_order

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the graph-defined network.

            Args:
                x: Input tensor, shape (batch_size, input_dim)

            Returns:
                output: Output tensor, shape (batch_size, output_dim)
            """
            batch_size = x.shape[0]

            if self.layer_order is not None:
                # Process layer by layer, building up activation lists
                layer_activations = []

                # Input layer
                input_nodes = self.layer_order[0]
                layer_activations.append(x)

                # Process each hidden/output layer
                for layer_idx in range(1, len(self.layer_order)):
                    layer_nodes = self.layer_order[layer_idx]
                    layer_size = len(layer_nodes)

                    # Compute activations for all nodes in this layer at once
                    # Build weight matrix slice for this layer (all_inputs x layer_nodes)
                    layer_weights = self.weight_matrix[:, layer_nodes]  # (n_nodes, layer_size)
                    layer_biases = self.bias[layer_nodes]  # (layer_size,)

                    # Get all activations computed so far (stack previous layers)
                    all_acts = torch.zeros(batch_size, self.n_nodes, device=self.device, dtype=x.dtype)
                    for i, nodes in enumerate(self.layer_order[:layer_idx]):
                        all_acts[:, nodes] = layer_activations[i]

                    # Compute z for all nodes in layer: (batch, n_nodes) @ (n_nodes, layer_size)
                    z = torch.matmul(all_acts, layer_weights) + layer_biases  # (batch, layer_size)
                    layer_output = self._activation(z)
                    layer_activations.append(layer_output)

                # Return final layer output
                return layer_activations[-1]
            else:
                raise NotImplementedError("Only layered graphs supported currently")

        def _activation(self, x: torch.Tensor) -> torch.Tensor:
            """Apply activation function."""
            if self.activation_name == 'tanh':
                return torch.tanh(x)
            elif self.activation_name == 'relu':
                return F.relu(x)
            elif self.activation_name == 'sigmoid':
                return torch.sigmoid(x)
            elif self.activation_name == 'linear':
                return x
            else:
                return torch.tanh(x)

        def sync_to_neural_graph(self):
            """Copy trained weights and biases back to NeuralGraph."""
            W = self.weight_matrix.detach().cpu().numpy()
            self.neural_graph.set_weight_matrix(W)
            self.neural_graph.biases = self.bias.detach().cpu().numpy()

        def sync_from_neural_graph(self):
            """Copy weights and biases from NeuralGraph to PyTorch."""
            W = self.neural_graph.get_weight_matrix()
            self.weight_matrix.data = torch.from_numpy(W).to(self.device)
            self.bias.data = torch.from_numpy(self.neural_graph.biases).to(self.device)


    def to_pytorch(
        neural_graph: 'NeuralGraph',
        device: Optional[str] = None
    ) -> 'GraphDefinedNetwork':
        """
        Convert NeuralGraph to PyTorch module for training.

        Args:
            neural_graph: Source neural graph (defines network architecture)
            device: Target device ('cuda', 'cpu', or None for auto)

        Returns:
            pytorch_model: PyTorch nn.Module

        Example:
            >>> from nekos.ai import LayerBuilder, NeuralGraph, to_pytorch
            >>> # Create sparse network
            >>> graph, clustering = LayerBuilder.sparse_mlp([784, 128, 10], connection_prob=0.3)
            >>> ng = NeuralGraph(graph, clustering)
            >>>
            >>> # Train with PyTorch
            >>> model = to_pytorch(ng)
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        return GraphDefinedNetwork(neural_graph, device)


    def from_pytorch(
        pytorch_model: 'GraphDefinedNetwork',
        neural_graph: Optional['NeuralGraph'] = None
    ) -> 'NeuralGraph':
        """
        Sync trained PyTorch weights back to NeuralGraph.

        Args:
            pytorch_model: Trained PyTorch model
            neural_graph: Target neural graph (or use model's internal reference)

        Returns:
            neural_graph: Updated NeuralGraph with trained weights
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        if neural_graph is None:
            neural_graph = pytorch_model.neural_graph

        pytorch_model.sync_to_neural_graph()
        return neural_graph


    def train_with_pytorch(
        neural_graph: 'NeuralGraph',
        train_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_name: str = 'adam',
        loss_fn: str = 'mse',
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train a graph-defined neural network using PyTorch backend.

        This trains networks with exotic architectures defined by graph structure:
        - Sparse random connectivity
        - Small-world networks
        - Skip connections
        - Custom connectivity patterns

        Args:
            neural_graph: Neural graph to train (architecture defined by graph)
            train_data: (X_train, y_train) tuple of numpy arrays
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            optimizer_name: 'adam', 'sgd', 'adamw'
            loss_fn: 'mse' or 'cross_entropy'
            validation_data: Optional (X_val, y_val)
            device: 'cuda', 'cpu', or None
            verbose: Print training progress

        Returns:
            history: Dict with 'train_loss', 'val_loss', 'epochs'

        Example:
            >>> from nekos.ai import LayerBuilder, NeuralGraph, train_with_pytorch
            >>>
            >>> # Create a sparse network (30% connectivity)
            >>> graph, clustering = LayerBuilder.sparse_mlp([784, 256, 128, 10],
            ...                                               connection_prob=0.3)
            >>> ng = NeuralGraph(graph, clustering, activation_fn='relu')
            >>>
            >>> # Train it
            >>> history = train_with_pytorch(
            ...     ng,
            ...     train_data=(X_train, y_train),
            ...     validation_data=(X_val, y_val),
            ...     epochs=50,
            ...     learning_rate=0.001
            ... )
            >>>
            >>> # Visualize the trained network
            >>> ng.graph.visualize(layout='force', clustering=ng.clustering)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        # Convert to PyTorch
        model = to_pytorch(neural_graph, device)

        # Prepare data
        X_train, y_train = train_data
        X_train_t = torch.from_numpy(X_train).float().to(model.device)
        y_train_t = torch.from_numpy(y_train).float().to(model.device)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_t = torch.from_numpy(X_val).float().to(model.device)
            y_val_t = torch.from_numpy(y_val).float().to(model.device)

        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup loss function
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }

        # Training loop
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            # Shuffle data
            perm = torch.randperm(n_samples)
            X_train_shuffled = X_train_t[perm]
            y_train_shuffled = y_train_t[perm]

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)
            history['epochs'].append(epoch)

            # Validation
            if validation_data is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    history['val_loss'].append(val_loss)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                if validation_data is not None:
                    print(f"Epoch {epoch}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}")

        # Sync weights back to neural graph
        from_pytorch(model, neural_graph)

        return history

else:
    # Dummy implementations when PyTorch not available
    def to_pytorch(*args, **kwargs):
        raise ImportError("PyTorch not available. Install with: pip install torch")

    def from_pytorch(*args, **kwargs):
        raise ImportError("PyTorch not available. Install with: pip install torch")

    def train_with_pytorch(*args, **kwargs):
        raise ImportError("PyTorch not available. Install with: pip install torch")
