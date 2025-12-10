"""
Graph visualization using vispy (OpenGL-accelerated)
"""
import numpy as np

def visualize(graph, layout='force', iterations=50, node_size=5, edge_width=1,
              node_color=(0.3, 0.7, 1.0, 1.0), edge_color=(0.5, 0.5, 0.5, 0.3),
              background_color=(0.0, 0.0, 0.0, 1.0),
              num_threads=1, verbose=False):
    """
    Visualize graph using vispy (GPU-accelerated with OpenGL)

    Parameters
    ----------
    graph : Graph
        The graph to visualize
    layout : str
        Layout algorithm ('force' for force-directed, 'random' for random)
    iterations : int
        Number of iterations for force-directed layout
    node_size : float
        Size of nodes in pixels
    edge_width : float
        Width of edges in pixels
    node_color : tuple
        RGBA color for nodes (values 0-1)
    edge_color : tuple
        RGBA color for edges (values 0-1)
    background_color : tuple
        RGBA color for background (values 0-1)
    num_threads : int
        Number of threads for parallel layout computation
    verbose : bool
        Print progress information
    """
    try:
        from vispy import app, scene
        from vispy.scene import visuals
    except ImportError:
        raise ImportError(
            "vispy is required for visualization. Install with: pip install vispy"
        )

    n = graph.num_nodes()
    if n == 0:
        print("Graph is empty, nothing to visualize")
        return

    # Generate layout using fast C++ implementation
    pos_list = graph.compute_layout(layout=layout, iterations=iterations,
                                    num_threads=num_threads, verbose=verbose)

    # Convert to numpy array
    pos = np.array(pos_list, dtype=np.float32)

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

    # Draw edges first (so they appear behind nodes)
    if edge_pos is not None:
        line_visual = visuals.Line(pos=edge_pos, color=edge_color,
                                   width=edge_width, connect='segments',
                                   method='gl', parent=view.scene)

    # Draw nodes
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color='white', face_color=node_color,
                    size=node_size, edge_width=0)
    view.add(scatter)

    # Start the application
    app.run()
