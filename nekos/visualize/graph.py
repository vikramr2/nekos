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
              color_by_cluster=True,
              collapsed=False,
              max_node_size=50,
              max_edge_width=10):
    """
    Visualize graph using vispy (GPU-accelerated with OpenGL)

    Parameters
    ----------
    graph : Graph
        The graph to visualize
    layout : str
        Layout algorithm ('force' for force-directed, 'random' for random,
        'clustered' for collapsed cluster view where each cluster is a single node)
    iterations : int
        Number of iterations for force-directed layout
    node_size : float
        Size of nodes in pixels (minimum size when collapsed=True)
    edge_width : float
        Width of edges in pixels (minimum width when collapsed=True)
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
        Clustering object (required for layout='clustered' or collapsed=True)
    cluster_separation : float
        Not used in collapsed view (kept for backward compatibility)
    intra_cluster_strength : float
        Not used in collapsed view (kept for backward compatibility)
    color_by_cluster : bool
        If True and clustering is provided, color nodes by their cluster (default: True)
    collapsed : bool
        If True, show collapsed view where each cluster is a single node (requires clustering)
    max_node_size : float
        Maximum node size in pixels for largest cluster (only used when collapsed=True)
    max_edge_width : float
        Maximum edge width in pixels for edge with most connections (only used when collapsed=True)
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

    # Handle collapsed view (automatic for 'clustered' layout or explicit collapsed=True)
    if collapsed or layout == 'clustered':
        if clustering is None:
            raise ValueError("clustering parameter is required when layout='clustered' or collapsed=True")

        # Get cluster IDs
        cluster_ids = clustering.get_cluster_ids()
        n_clusters = len(cluster_ids)

        if n_clusters == 0:
            print("No clusters found, nothing to visualize")
            return

        if verbose:
            print(f"Building collapsed graph with {n_clusters} clusters...")

        # Build cluster size map
        cluster_sizes = {}
        for cluster_id in cluster_ids:
            nodes = clustering.get_cluster_nodes(cluster_id)
            cluster_sizes[cluster_id] = len(nodes)

        # Build inter-cluster edge counts
        inter_cluster_edges = {}

        edges = graph.get_edges()
        for u, v in edges:
            # Get original IDs
            u_orig = graph.get_original_id(u)
            v_orig = graph.get_original_id(v)

            # Get cluster IDs using the clustering's method
            u_cluster = clustering.get_node_cluster_by_original_id(u_orig)
            v_cluster = clustering.get_node_cluster_by_original_id(v_orig)

            # Skip if either node is not in a cluster
            if u_cluster is None or v_cluster is None:
                continue

            # Skip intra-cluster edges
            if u_cluster == v_cluster:
                continue

            # Count inter-cluster edge (ensure consistent ordering)
            edge_key = tuple(sorted([u_cluster, v_cluster]))
            inter_cluster_edges[edge_key] = inter_cluster_edges.get(edge_key, 0) + 1

        if verbose:
            print(f"Found {len(inter_cluster_edges)} inter-cluster edges")

        # Create a temporary graph with clusters as nodes
        import nekos
        cluster_graph = nekos.Graph()

        # Map cluster IDs to internal node indices
        # Cluster IDs are strings, so we use enumerate index as the node ID
        cluster_id_to_idx = {}
        for i, cluster_id in enumerate(cluster_ids):
            # Use integer index as the original_id for the temporary graph
            internal_idx = cluster_graph.add_node(i)
            cluster_id_to_idx[cluster_id] = internal_idx

        # Add edges between clusters
        for (c1, c2), weight in inter_cluster_edges.items():
            idx1 = cluster_id_to_idx[c1]
            idx2 = cluster_id_to_idx[c2]
            cluster_graph.add_edge(idx1, idx2)

        # Generate layout using the specified layout algorithm (ignore clustered-specific params)
        layout_type = layout if layout != 'clustered' else 'force'
        pos_list = cluster_graph.compute_layout(layout=layout_type, iterations=iterations,
                                                num_threads=num_threads, verbose=verbose)
        pos = np.array(pos_list, dtype=np.float32)

        if verbose:
            print(f"Position range: x=[{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}], y=[{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")

        # Calculate node sizes based on cluster sizes
        min_cluster_size = min(cluster_sizes.values())
        max_cluster_size = max(cluster_sizes.values())

        node_sizes = np.zeros(n_clusters, dtype=np.float32)
        for i, cluster_id in enumerate(cluster_ids):
            size = cluster_sizes[cluster_id]
            # Scale linearly between min and max node size
            if max_cluster_size > min_cluster_size:
                normalized = (size - min_cluster_size) / (max_cluster_size - min_cluster_size)
                node_sizes[i] = node_size + normalized * (max_node_size - node_size)
            else:
                node_sizes[i] = node_size

        # Generate colors for clusters
        cluster_colors = generate_cluster_colors(cluster_ids, alpha=node_alpha if node_alpha else 1.0)
        node_colors = np.array([cluster_colors[cid] for cid in cluster_ids], dtype=np.float32)

        # Prepare edge visualization with varying widths
        if len(inter_cluster_edges) == 0:
            print("No inter-cluster edges found, showing only nodes")
            edge_pos = None
            edge_widths = None
        else:
            edge_pos = []
            edge_widths = []

            min_edge_weight = min(inter_cluster_edges.values())
            max_edge_weight = max(inter_cluster_edges.values())

            for (c1, c2), weight in inter_cluster_edges.items():
                idx1 = cluster_id_to_idx[c1]
                idx2 = cluster_id_to_idx[c2]

                edge_pos.append(pos[idx1])
                edge_pos.append(pos[idx2])

                # Scale edge width based on weight
                if max_edge_weight > min_edge_weight:
                    normalized = (weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
                    width = edge_width + normalized * (max_edge_width - edge_width)
                else:
                    width = edge_width

                edge_widths.append(width)

            edge_pos = np.array(edge_pos, dtype=np.float32)

        # Create canvas - don't show immediately to allow setup to complete
        canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor=background_color,
                                  title=f'Collapsed Graph (clusters={n_clusters}, inter-cluster edges={len(inter_cluster_edges)})')
        view = canvas.central_widget.add_view()
        view.camera = scene.PanZoomCamera(aspect=1)

        # Draw nodes
        scatter = visuals.Markers()
        scatter.set_data(pos, edge_color='white', face_color=node_colors,
                        size=node_sizes, edge_width=0)
        view.add(scatter)

        # Draw edges - use single Line visual for better performance
        if edge_pos is not None:
            # For now, use a uniform width (average of all widths) for performance
            # Drawing thousands of separate Line visuals is too slow
            avg_width = np.mean(edge_widths)
            line_visual = visuals.Line(pos=edge_pos, color=edge_color,
                                       width=avg_width, connect='segments',
                                       method='gl', parent=view.scene)
            if verbose:
                print(f"Drawing {len(inter_cluster_edges)} edges with average width {avg_width:.2f}")

        # Set camera range after all visuals are added
        view.camera.set_range()

        if verbose:
            print(f"Cluster sizes: min={min_cluster_size}, max={max_cluster_size}")
            if len(inter_cluster_edges) > 0:
                print(f"Edge weights: min={min_edge_weight}, max={max_edge_weight}")
            print("Opening visualization window... (close window to continue)")

        # Show the canvas after all setup is complete
        canvas.show()

        # Start the application
        app.run()

        if verbose:
            print("Visualization window closed")
        return

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
