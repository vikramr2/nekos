"""
Graph visualization using vispy (OpenGL-accelerated)
"""
import numpy as np
import colorsys

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert hex color string to RGBA tuple

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., '#RRGGBB' or 'RRGGBB', '#AAA', 'AAA')
    alpha : float
        Alpha value (0.0 to 1.0)

    Returns
    -------
    tuple
        RGBA color tuple with values in range [0, 1]
    """
    if not alpha:
        alpha = 1.0

    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        raise ValueError("Hex color must be in format RRGGBB")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

def generate_cluster_colors(cluster_ids, alpha=1.0):
    """
    Generate distinct colors for each cluster using HSV color space

    Parameters
    ----------
    cluster_ids : list
        List of cluster IDs
    alpha : float
        Alpha value for all colors

    Returns
    -------
    dict
        Mapping from cluster_id to RGBA tuple
    """
    n_clusters = len(cluster_ids)
    colors = {}

    for i, cluster_id in enumerate(cluster_ids):
        # Distribute hues evenly around the color wheel
        hue = i / max(n_clusters, 1)
        # Use high saturation and value for vibrant colors
        saturation = 0.8
        value = 0.9

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[cluster_id] = (r, g, b, alpha)

    return colors

def visualize(graph,
              layout='force',
              iterations=50,
              node_size=5,
              edge_width=1,
              node_color=(1.0, 1.0, 1.0, 1.0),
              edge_color=(0.8, 0.8, 0.8, 1.0),
              background_color=(0.0, 0.0, 0.0, 1.0),
              num_threads=1,
              node_alpha=None,
              edge_alpha=None,
              background_alpha=None,
              verbose=False,
              clustering=None,
              cluster_separation=2.0,
              intra_cluster_strength=1.0,
              color_by_cluster=True):
    """
    Visualize graph using vispy (GPU-accelerated with OpenGL)

    Parameters
    ----------
    graph : Graph
        The graph to visualize
    layout : str
        Layout algorithm ('force' for force-directed, 'random' for random, 'clustered' for cluster-aware layout)
    iterations : int
        Number of iterations for force-directed layout
    node_size : float
        Size of nodes in pixels
    edge_width : float
        Width of edges in pixels
    node_color : tuple
        RGBA color for nodes (values 0-1). Ignored if color_by_cluster=True and clustering is provided.
    edge_color : tuple
        RGBA color for edges (values 0-1)
    background_color : tuple
        RGBA color for background (values 0-1)
    num_threads : int
        Number of threads for parallel layout computation
    verbose : bool
        Print progress information
    clustering : Clustering, optional
        Clustering object (required for layout='clustered')
    cluster_separation : float
        Separation strength between clusters (for layout='clustered')
    intra_cluster_strength : float
        Attraction strength within clusters (for layout='clustered')
    color_by_cluster : bool
        If True and clustering is provided, color nodes by their cluster (default: True)
    """
    try:
        from vispy import app, scene
        from vispy.scene import visuals
    except ImportError:
        raise ImportError(
            "vispy is required for visualization. Install with: pip install vispy"
        )
    
    # Check if edge/background colors are hex strings and convert to RGBA
    if isinstance(edge_color, str):
        edge_color = hex_to_rgba(edge_color, edge_alpha if edge_alpha is not None else 1.0)
    if isinstance(background_color, str):
        background_color = hex_to_rgba(background_color, background_alpha if background_alpha is not None else 1.0)

    # Get number of nodes
    n = graph.num_nodes()
    if n == 0:
        print("Graph is empty, nothing to visualize")
        return

    # Generate layout using fast C++ implementation
    if layout == 'clustered':
        if clustering is None:
            raise ValueError("clustering parameter is required when layout='clustered'")
        pos_list = graph.compute_clustered_layout(
            clustering,
            iterations=iterations,
            num_threads=num_threads,
            verbose=verbose,
            cluster_separation=cluster_separation,
            intra_cluster_strength=intra_cluster_strength
        )
    else:
        pos_list = graph.compute_layout(layout=layout, iterations=iterations,
                                        num_threads=num_threads, verbose=verbose)

    # Convert to numpy array
    pos = np.array(pos_list, dtype=np.float32)

    # Color nodes by cluster if requested
    if color_by_cluster and clustering is not None:
        # Get cluster colors
        cluster_ids = clustering.get_cluster_ids()
        cluster_colors = generate_cluster_colors(cluster_ids, alpha=node_alpha if node_alpha else 1.0)

        # Create color array for each node
        node_colors = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            # Get original ID for this internal node
            original_id = graph.get_original_id(i)
            cluster_id = clustering.get_node_cluster_by_original_id(original_id)

            if cluster_id and cluster_id in cluster_colors:
                node_colors[i] = cluster_colors[cluster_id]
            else:
                # Node not in any cluster - use gray
                node_colors[i] = (0.5, 0.5, 0.5, node_alpha if node_alpha else 1.0)

        # Override node_color with the array
        node_color = node_colors
    elif isinstance(node_color, str):
        # Convert hex string to RGBA if needed
        node_color = hex_to_rgba(node_color, node_alpha if node_alpha is not None else 1.0)

    # Get edges
    edges_list = graph.get_edges()
    if len(edges_list) == 0:
        print("Graph has no edges, showing only nodes")
        edge_pos = None
    else:
        # Convert edges to numpy array for vispy
        edge_pos = np.zeros((len(edges_list) * 2, 2), dtype=np.float32)
        for i, (u, v) in enumerate(edges_list):
            edge_pos[2*i] = pos[u]
            edge_pos[2*i + 1] = pos[v]

    # Create canvas with background color
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=background_color,
                              title=f'Graph Visualization (nodes={n}, edges={len(edges_list)})')
    view = canvas.central_widget.add_view()

    # Use PanZoomCamera for zooming and panning
    # PanZoomCamera natively supports:
    #   - Mouse wheel zoom
    #   - Trackpad pinch-to-zoom (two-finger pinch)
    #   - Click and drag to pan
    #   - Two-finger drag to pan (trackpad)
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range()

    # Draw nodes first
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color='white', face_color=node_color,
                    size=node_size, edge_width=0)
    view.add(scatter)

    # Draw edges last (so they appear on top of nodes)
    if edge_pos is not None:
        line_visual = visuals.Line(pos=edge_pos, color=edge_color,
                                   width=edge_width, connect='segments',
                                   method='gl', parent=view.scene)

    # Start the application
    app.run()
