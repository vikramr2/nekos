"""
Visualization tools for neural graphs - see activations, gradients, and weights
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
# import warnings


def visualize_activations(
    neural_graph: 'NeuralGraph',
    inputs: Optional[np.ndarray] = None,
    layout: str = 'force',
    iterations: int = 50,
    node_size_scale: float = 10.0,
    **kwargs
) -> None:
    """
    Visualize neural graph with node sizes proportional to activation magnitude.

    Args:
        neural_graph: Neural graph to visualize
        inputs: Optional input to run forward pass first
        layout: 'force', 'clustered', or 'random'
        iterations: Layout iterations
        node_size_scale: Scale factor for node sizes
        **kwargs: Additional arguments passed to graph.visualize()

    Example:
        >>> from nekos.ai import NeuralGraph, visualize_activations
        >>> ng = NeuralGraph(graph, clustering)
        >>> ng.forward(X_sample[0])
        >>> visualize_activations(ng, node_size_scale=15)
    """
    # Run forward pass if input provided
    if inputs is not None:
        neural_graph.forward(inputs)

    # Get activation magnitudes
    activations = np.abs(neural_graph.activations)

    # Normalize to [0, 1] range for alpha values
    max_activation = activations.max()
    if max_activation > 0:
        normalized_activations = activations / max_activation
    else:
        normalized_activations = activations

    # Use alpha channel to show activation magnitude (min 0.3 for visibility)
    node_alphas = normalized_activations * 0.7 + 0.3

    # Create visualization kwargs
    viz_kwargs = {
        'layout': layout,
        'iterations': iterations,
        'node_size': node_size_scale,
        'node_alpha': node_alphas.tolist(),
        **kwargs
    }

    # Add clustering if using clustered layout and layer_metadata available
    if layout == 'clustered' and hasattr(neural_graph, 'layer_metadata'):
        if 'clustering' not in viz_kwargs:
            # Create a simple clustering object from layer metadata if possible
            # For now, just inform the user
            print("Note: Clustered layout requires clustering object from nekos C++ API")
            viz_kwargs['layout'] = 'force'  # Fall back to force layout

    print(f"Visualizing activations (max={max_activation:.4f})")
    print("Node opacity = activation magnitude")

    neural_graph.graph.visualize(**viz_kwargs)


def visualize_gradients(
    neural_graph: 'NeuralGraph',
    inputs: np.ndarray,
    targets: np.ndarray,
    layout: str = 'force',
    iterations: int = 50,
    node_size_scale: float = 10.0,
    **kwargs
) -> None:
    """
    Visualize neural graph with node sizes proportional to gradient magnitude.

    Useful for understanding:
    - Which neurons receive strong learning signals
    - Dead/dying neurons (zero gradient)
    - Gradient flow through exotic architectures

    Args:
        neural_graph: Neural graph to visualize
        inputs: Input for forward pass
        targets: Targets for backward pass
        layout: 'force', 'clustered', or 'random'
        iterations: Layout iterations
        node_size_scale: Scale factor for node sizes
        **kwargs: Additional arguments passed to graph.visualize()

    Example:
        >>> from nekos.ai import visualize_gradients
        >>> visualize_gradients(ng, X_sample[0], y_sample[0])
    """
    # Run forward and backward pass
    neural_graph.forward(inputs)
    loss = neural_graph.backward(targets)

    # Get gradient magnitudes
    gradients = np.abs(neural_graph.gradients)

    # Normalize to [0, 1] range for alpha values
    max_gradient = gradients.max()
    if max_gradient > 0:
        normalized_gradients = gradients / max_gradient
    else:
        normalized_gradients = gradients

    # Use alpha channel to show gradient magnitude (min 0.3 for visibility)
    node_alphas = normalized_gradients * 0.7 + 0.3

    # Create visualization kwargs
    viz_kwargs = {
        'layout': layout,
        'iterations': iterations,
        'node_size': node_size_scale,
        'node_alpha': node_alphas.tolist(),
        **kwargs
    }

    print(f"Visualizing gradients (loss={loss:.4f}, max_grad={max_gradient:.4f})")
    print("Node opacity = gradient magnitude")
    print(f"Dead neurons (zero gradient): {(gradients == 0).sum()}/{len(gradients)}")

    neural_graph.graph.visualize(**viz_kwargs)


def visualize_weights(
    neural_graph: 'NeuralGraph',
    layout: str = 'force',
    iterations: int = 50,
    weight_scale: float = 5.0,
    show_weight_distribution: bool = True,
    **kwargs
) -> None:
    """
    Visualize neural graph with edge thickness proportional to weight magnitude.

    Note: Current nekos visualization doesn't support variable edge thickness,
    so this function shows weight statistics and node connectivity.

    Args:
        neural_graph: Neural graph to visualize
        layout: 'force', 'clustered', or 'random'
        iterations: Layout iterations
        weight_scale: Scale factor (for future edge thickness support)
        show_weight_distribution: Print weight statistics
        **kwargs: Additional arguments passed to graph.visualize()

    Example:
        >>> from nekos.ai import visualize_weights
        >>> visualize_weights(ng)
    """
    # Get weights
    weights = np.array(list(neural_graph.weights.values()))

    if show_weight_distribution:
        print("\n=== Weight Statistics ===")
        print(f"Total connections: {len(weights)}")
        print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"Mean weight: {weights.mean():.4f}")
        print(f"Std weight: {weights.std():.4f}")

        # Histogram
        print("\nWeight distribution:")
        hist, bins = np.histogram(weights, bins=10)
        for i in range(len(hist)):
            bar = '█' * int(hist[i] / hist.max() * 30)
            print(f"[{bins[i]:6.3f}, {bins[i+1]:6.3f}): {bar} ({hist[i]})")

    # Compute node connectivity (degree)
    degrees = np.zeros(neural_graph.graph.num_nodes())
    for node in range(neural_graph.graph.num_nodes()):
        degrees[node] = len(neural_graph.graph.neighbors(node))

    # Scale by degree for visualization
    node_sizes = (degrees / degrees.max() * 10.0 + 2.0).tolist()

    # Create visualization kwargs
    viz_kwargs = {
        'layout': layout,
        'iterations': iterations,
        'node_size': node_sizes,
        **kwargs
    }

    print("\nNode size = connectivity (degree)")
    neural_graph.graph.visualize(**viz_kwargs)


def visualize_learning(
    neural_graph: 'NeuralGraph',
    train_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 100,
    learning_rate: float = 0.01,
    visualize_every: int = 10,
    layout: str = 'force',
    iterations: int = 30,
    **kwargs
) -> None:
    """
    Visualize neural graph during training - watch the network learn!

    Shows network state periodically during training.
    Warning: This will be slow due to visualization overhead.

    Args:
        neural_graph: Neural graph to train and visualize
        train_data: (X_train, y_train)
        epochs: Number of training epochs
        learning_rate: Learning rate
        visualize_every: Visualize every N epochs
        layout: Layout type
        iterations: Layout iterations
        **kwargs: Additional visualization arguments

    Example:
        >>> from nekos.ai import visualize_learning
        >>> visualize_learning(ng, (X_train, y_train), epochs=50, visualize_every=5)
    """
    X_train, y_train = train_data
    n_samples = X_train.shape[0]

    print(f"Training for {epochs} epochs, visualizing every {visualize_every} epochs")
    print("Close visualization window to continue training...\n")

    for epoch in range(epochs):
        # Training epoch
        epoch_loss = 0.0
        for i in range(n_samples):
            loss = neural_graph.train_step(
                X_train[i],
                y_train[i],
                learning_rate=learning_rate
            )
            epoch_loss += loss

        avg_loss = epoch_loss / n_samples

        # Visualize periodically
        if epoch % visualize_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

            # Run forward pass on first sample
            neural_graph.forward(X_train[0])

            # Visualize activations
            visualize_activations(
                neural_graph,
                inputs=None,  # Already computed
                layout=layout,
                iterations=iterations,
                **kwargs
            )


def compare_architectures(
    neural_graphs: Dict[str, 'NeuralGraph'],
    inputs: np.ndarray,
    layout: str = 'force',
    iterations: int = 50,
    **kwargs
) -> None:
    """
    Compare multiple neural graph architectures side by side.

    Args:
        neural_graphs: Dict of {name: neural_graph}
        inputs: Input to propagate through networks
        layout: Layout type
        iterations: Layout iterations
        **kwargs: Visualization arguments

    Example:
        >>> mlp_graph, mlp_clustering = LayerBuilder.mlp([100, 50, 10])
        >>> sparse_graph, sparse_clustering = LayerBuilder.sparse_mlp([100, 50, 10], 0.3)
        >>>
        >>> mlp_ng = NeuralGraph(mlp_graph, mlp_clustering)
        >>> sparse_ng = NeuralGraph(sparse_graph, sparse_clustering)
        >>>
        >>> compare_architectures(
        ...     {'MLP': mlp_ng, 'Sparse MLP': sparse_ng},
        ...     inputs=X_sample[0]
        ... )
    """
    for name, neural_graph in neural_graphs.items():
        print(f"\n{'='*60}")
        print(f"Architecture: {name}")
        print(f"{neural_graph}")
        print(f"{'='*60}")

        neural_graph.forward(inputs)
        visualize_activations(
            neural_graph,
            inputs=None,
            layout=layout,
            iterations=iterations,
            **kwargs
        )


def architecture_stats(neural_graph: 'NeuralGraph') -> Dict[str, Any]:
    """
    Compute statistics about the neural graph architecture.

    Returns:
        stats: Dictionary with architecture statistics

    Example:
        >>> stats = architecture_stats(ng)
        >>> print(f"Sparsity: {stats['sparsity']:.2%}")
    """
    n_nodes = neural_graph.graph.num_nodes()
    n_edges = neural_graph.graph.num_edges()

    # Compute maximum possible edges (for feedforward network)
    if neural_graph.layer_order is not None:
        max_edges = 0
        for i in range(len(neural_graph.layer_order) - 1):
            src_size = len(neural_graph.layer_order[i])
            dst_size = len(neural_graph.layer_order[i + 1])
            max_edges += src_size * dst_size
    else:
        # For general graph, max is complete graph
        max_edges = n_nodes * (n_nodes - 1)

    sparsity = 1.0 - (n_edges / max_edges) if max_edges > 0 else 0.0

    # Compute degree statistics
    degrees = np.array([
        len(neural_graph.graph.neighbors(node))
        for node in range(n_nodes)
    ])

    # Weight statistics - get from C++ weight matrix
    weight_matrix = neural_graph.get_weight_matrix()
    weights = weight_matrix[weight_matrix != 0]  # Non-zero weights only

    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'max_possible_edges': max_edges,
        'sparsity': sparsity,
        'avg_degree': degrees.mean(),
        'max_degree': degrees.max(),
        'min_degree': degrees.min(),
        'weight_mean': weights.mean(),
        'weight_std': weights.std(),
        'weight_min': weights.min(),
        'weight_max': weights.max(),
    }

    if neural_graph.layer_order is not None:
        stats['n_layers'] = len(neural_graph.layer_order)
        stats['layer_sizes'] = [len(layer) for layer in neural_graph.layer_order]

    return stats


def print_architecture_stats(neural_graph: 'NeuralGraph') -> None:
    """
    Print formatted architecture statistics.

    Example:
        >>> from nekos.ai import print_architecture_stats
        >>> print_architecture_stats(ng)
    """
    stats = architecture_stats(neural_graph)

    print("\n" + "="*60)
    print("Neural Graph Architecture Statistics")
    print("="*60)

    print(f"\nTopology:")
    print(f"  Nodes: {stats['n_nodes']}")
    print(f"  Connections: {stats['n_edges']}")
    print(f"  Max possible: {stats['max_possible_edges']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")

    print(f"\nConnectivity:")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    print(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")

    print(f"\nWeights:")
    print(f"  Mean: {stats['weight_mean']:.4f} ± {stats['weight_std']:.4f}")
    print(f"  Range: [{stats['weight_min']:.4f}, {stats['weight_max']:.4f}]")

    if 'n_layers' in stats:
        print(f"\nLayers:")
        print(f"  Number of layers: {stats['n_layers']}")
        print(f"  Layer sizes: {stats['layer_sizes']}")

    print("="*60 + "\n")
