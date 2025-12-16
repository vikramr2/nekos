"""
LayerBuilder - Utility for constructing neural graphs with various topologies
"""

import numpy as np
import nekos
from typing import List, Tuple, Optional, Dict, Any


class LayerBuilder:
    """
    Build neural graphs with various connectivity patterns.

    Supports:
    - Standard MLP (fully connected layers)
    - Sparse random connectivity
    - Small-world networks
    - Skip connections
    - Custom connectivity patterns
    """

    def __init__(self):
        """Initialize builder."""
        self.edges = []
        self.node_to_layer = {}
        self.layer_sizes = []
        self.current_node_id = 0

    def add_layer(self, size: int) -> List[int]:
        """
        Add a layer with specified number of nodes.

        Args:
            size: Number of nodes in this layer

        Returns:
            node_ids: List of node IDs in this layer
        """
        layer_idx = len(self.layer_sizes)
        self.layer_sizes.append(size)

        node_ids = []
        for _ in range(size):
            self.node_to_layer[self.current_node_id] = layer_idx
            node_ids.append(self.current_node_id)
            self.current_node_id += 1

        return node_ids

    def connect_layers_full(
        self,
        src_layer: List[int],
        dst_layer: List[int]
    ):
        """
        Fully connect two layers (standard MLP connectivity).

        Args:
            src_layer: Source layer node IDs
            dst_layer: Destination layer node IDs
        """
        for src in src_layer:
            for dst in dst_layer:
                self.edges.append((src, dst))

    def connect_layers_sparse(
        self,
        src_layer: List[int],
        dst_layer: List[int],
        connection_prob: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Sparsely connect two layers with random dropout.

        Args:
            src_layer: Source layer node IDs
            dst_layer: Destination layer node IDs
            connection_prob: Probability of creating each connection
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        for src in src_layer:
            for dst in dst_layer:
                if np.random.rand() < connection_prob:
                    self.edges.append((src, dst))

    def connect_layers_random_k(
        self,
        src_layer: List[int],
        dst_layer: List[int],
        k: int,
        seed: Optional[int] = None
    ):
        """
        Connect each destination node to k random source nodes.

        Args:
            src_layer: Source layer node IDs
            dst_layer: Destination layer node IDs
            k: Number of incoming connections per dst node
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        for dst in dst_layer:
            sources = np.random.choice(src_layer, size=min(k, len(src_layer)), replace=False)
            for src in sources:
                self.edges.append((src, dst))

    def add_skip_connections(
        self,
        src_layer: List[int],
        dst_layer: List[int],
        skip_type: str = 'one_to_one'
    ):
        """
        Add skip connections between non-adjacent layers.

        Args:
            src_layer: Source layer
            dst_layer: Destination layer
            skip_type: 'one_to_one' (same index) or 'full' (all connections)
        """
        if skip_type == 'one_to_one':
            for i in range(min(len(src_layer), len(dst_layer))):
                self.edges.append((src_layer[i], dst_layer[i]))
        elif skip_type == 'full':
            self.connect_layers_full(src_layer, dst_layer)
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")

    def add_recurrent_connections(
        self,
        layer: List[int],
        recurrent_prob: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Add recurrent connections within a layer.

        Args:
            layer: Layer node IDs
            recurrent_prob: Probability of recurrent connection
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        for src in layer:
            for dst in layer:
                if src != dst and np.random.rand() < recurrent_prob:
                    self.edges.append((src, dst))

    def build(self) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build the final graph and layer metadata.

        Returns:
            graph: nekos.Graph with all connections
            layer_metadata: Dict with layer information (use this instead of Clustering)
        """
        # Create graph
        graph = nekos.Graph()

        # Add all nodes
        for node_id in range(self.current_node_id):
            graph.add_node(node_id)

        # Add all edges (undirected in nekos, but we'll track direction via layers)
        for src, dst in self.edges:
            graph.add_edge(src, dst)

        # Create layer metadata (replaces Clustering for neural networks)
        layer_metadata = {
            'node_to_layer': self.node_to_layer.copy(),
            'layer_sizes': self.layer_sizes.copy(),
            'layer_order': []
        }

        # Build layer_order (list of lists of node IDs)
        for layer_idx in range(len(self.layer_sizes)):
            layer_nodes = [node_id for node_id, l_idx in self.node_to_layer.items() if l_idx == layer_idx]
            layer_metadata['layer_order'].append(layer_nodes)

        return graph, layer_metadata

    @staticmethod
    def mlp(layer_sizes: List[int]) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build a standard MLP with fully connected layers.

        Args:
            layer_sizes: Number of nodes per layer, e.g., [784, 128, 64, 10]

        Returns:
            graph: Neural graph structure
            layer_metadata: Layer information dict

        Example:
            >>> graph, layers = LayerBuilder.mlp([784, 128, 10])
        """
        builder = LayerBuilder()

        layers = []
        for size in layer_sizes:
            layers.append(builder.add_layer(size))

        # Connect consecutive layers
        for i in range(len(layers) - 1):
            builder.connect_layers_full(layers[i], layers[i + 1])

        return builder.build()

    @staticmethod
    def sparse_mlp(
        layer_sizes: List[int],
        connection_prob: float = 0.3,
        seed: Optional[int] = None
    ) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build an MLP with sparse random connections.

        Args:
            layer_sizes: Number of nodes per layer
            connection_prob: Probability of each connection existing
            seed: Random seed

        Returns:
            graph, clustering
        """
        builder = LayerBuilder()

        layers = []
        for size in layer_sizes:
            layers.append(builder.add_layer(size))

        for i in range(len(layers) - 1):
            builder.connect_layers_sparse(
                layers[i], layers[i + 1],
                connection_prob=connection_prob,
                seed=seed
            )

        return builder.build()

    @staticmethod
    def resnet_like(
        layer_sizes: List[int],
        skip_frequency: int = 2
    ) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build a ResNet-like architecture with skip connections.

        Args:
            layer_sizes: Number of nodes per layer
            skip_frequency: Add skip connection every N layers

        Returns:
            graph, layer_metadata
        """
        builder = LayerBuilder()

        layers = []
        for size in layer_sizes:
            layers.append(builder.add_layer(size))

        # Connect consecutive layers
        for i in range(len(layers) - 1):
            builder.connect_layers_full(layers[i], layers[i + 1])

        # Add skip connections
        for i in range(len(layers) - skip_frequency):
            if (i + skip_frequency) < len(layers):
                builder.add_skip_connections(
                    layers[i],
                    layers[i + skip_frequency],
                    skip_type='one_to_one'
                )

        return builder.build()

    @staticmethod
    def small_world(
        layer_sizes: List[int],
        k_neighbors: int = 5,
        rewire_prob: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build a small-world neural network.

        Each layer is initially connected to k nearest neighbors,
        then connections are randomly rewired.

        Args:
            layer_sizes: Nodes per layer
            k_neighbors: Initial neighborhood size
            rewire_prob: Probability of rewiring each edge
            seed: Random seed

        Returns:
            graph, layer_metadata
        """
        if seed is not None:
            np.random.seed(seed)

        builder = LayerBuilder()

        layers = []
        for size in layer_sizes:
            layers.append(builder.add_layer(size))

        # Connect consecutive layers with k-nearest-neighbor + rewiring
        for i in range(len(layers) - 1):
            src_layer = layers[i]
            dst_layer = layers[i + 1]

            for dst_idx, dst in enumerate(dst_layer):
                # Connect to k nearest neighbors in src layer
                for k in range(k_neighbors):
                    src_idx = (dst_idx + k) % len(src_layer)
                    src = src_layer[src_idx]

                    # Rewire with probability
                    if np.random.rand() < rewire_prob:
                        src = np.random.choice(src_layer)

                    builder.edges.append((src, dst))

        return builder.build()

    @staticmethod
    def random_dag(
        n_nodes: int,
        edge_prob: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[nekos.Graph, Dict[str, Any]]:
        """
        Build a random directed acyclic graph (DAG).

        Nodes are topologically ordered, edges only go forward.

        Args:
            n_nodes: Total number of nodes
            edge_prob: Probability of edge between any valid pair
            seed: Random seed

        Returns:
            graph, layer_metadata (all nodes in one layer)
        """
        if seed is not None:
            np.random.seed(seed)

        builder = LayerBuilder()
        nodes = builder.add_layer(n_nodes)

        # Add edges only from lower to higher indices (ensures DAG)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < edge_prob:
                    builder.edges.append((nodes[i], nodes[j]))

        return builder.build()
