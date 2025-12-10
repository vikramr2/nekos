"""
Conversion utilities for nekos graphs to/from other graph libraries.
"""

def from_networkx(G):
    """
    Convert a NetworkX graph to a nekos Graph.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph (undirected)

    Returns
    -------
    nekos.Graph
        A nekos graph with the same structure

    Examples
    --------
    >>> import networkx as nx
    >>> import nekos
    >>> G = nx.karate_club_graph()
    >>> g = nekos.from_networkx(G)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function. Install it with: pip install networkx")

    from . import Graph

    if G.is_directed():
        raise ValueError("Only undirected graphs are supported. Convert with G.to_undirected() first.")

    # Create nekos graph
    g = Graph()

    # Map NetworkX node IDs to nekos internal IDs
    node_map = {}
    for node in G.nodes():
        # Convert node to integer if it isn't already
        node_id = int(node) if isinstance(node, (int, float)) else hash(node)
        internal_id = g.add_node(node_id)
        node_map[node] = internal_id

    # Add edges
    for u, v in G.edges():
        g.add_edge(node_map[u], node_map[v])

    return g


def to_networkx(g):
    """
    Convert a nekos Graph to a NetworkX graph.

    Parameters
    ----------
    g : nekos.Graph
        A nekos graph

    Returns
    -------
    networkx.Graph
        A NetworkX graph with the same structure

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_tsv("graph.tsv")
    >>> G = nekos.to_networkx(g)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for this function. Install it with: pip install networkx")

    # Create NetworkX graph
    G = nx.Graph()

    # Get original node IDs
    original_ids = list(g.get_original_ids())

    # Add nodes with original IDs
    G.add_nodes_from(original_ids)

    # Add edges using original IDs
    for internal_u, internal_v in g.get_edges():
        u = original_ids[internal_u]
        v = original_ids[internal_v]
        G.add_edge(u, v)

    return G


def from_networkit(G):
    """
    Convert a NetworKit graph to a nekos Graph.

    Parameters
    ----------
    G : networkit.Graph
        A NetworKit graph (undirected)

    Returns
    -------
    nekos.Graph
        A nekos graph with the same structure

    Examples
    --------
    >>> import networkit as nk
    >>> import nekos
    >>> G = nk.readGraph("graph.txt", nk.Format.EdgeListTabZero)
    >>> g = nekos.from_networkit(G)
    """
    try:
        import networkit as nk
    except ImportError:
        raise ImportError("NetworKit is required for this function. Install it with: pip install networkit")

    from . import Graph

    if G.isDirected():
        raise ValueError("Only undirected graphs are supported.")

    # Create nekos graph
    g = Graph()

    # Add nodes (NetworKit uses consecutive node IDs starting from 0)
    for node in range(G.numberOfNodes()):
        g.add_node(node)

    # Add edges
    for u, v in G.iterEdges():
        g.add_edge(u, v)

    return g


def to_networkit(g):
    """
    Convert a nekos Graph to a NetworKit graph.

    Parameters
    ----------
    g : nekos.Graph
        A nekos graph

    Returns
    -------
    networkit.Graph
        A NetworKit graph with the same structure

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_tsv("graph.tsv")
    >>> G = nekos.to_networkit(g)
    """
    try:
        import networkit as nk
    except ImportError:
        raise ImportError("NetworKit is required for this function. Install it with: pip install networkit")

    # Create NetworKit graph
    n = g.num_nodes()
    G = nk.Graph(n, weighted=False, directed=False)

    # Add edges (using internal node IDs which are 0-indexed)
    for internal_u, internal_v in g.get_edges():
        G.addEdge(internal_u, internal_v)

    return G


def from_igraph(G):
    """
    Convert an igraph Graph to a nekos Graph.

    Parameters
    ----------
    G : igraph.Graph
        An igraph graph (undirected)

    Returns
    -------
    nekos.Graph
        A nekos graph with the same structure

    Examples
    --------
    >>> import igraph as ig
    >>> import nekos
    >>> G = ig.Graph.Famous("Zachary")
    >>> g = nekos.from_igraph(G)
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("igraph is required for this function. Install it with: pip install igraph")

    from . import Graph

    if G.is_directed():
        raise ValueError("Only undirected graphs are supported. Convert with G.as_undirected() first.")

    # Create nekos graph
    g = Graph()

    # Add nodes
    # Check if vertices have 'name' attribute
    if 'name' in G.vs.attributes():
        for v in G.vs:
            # Use name if available, otherwise use vertex index
            node_id = int(v['name']) if isinstance(v['name'], (int, float)) else hash(v['name'])
            g.add_node(node_id)
    else:
        # Use vertex indices as node IDs
        for i in range(G.vcount()):
            g.add_node(i)

    # Add edges (igraph uses vertex indices)
    for edge in G.es:
        u, v = edge.tuple
        g.add_edge(u, v)

    return g


def to_igraph(g):
    """
    Convert a nekos Graph to an igraph Graph.

    Parameters
    ----------
    g : nekos.Graph
        A nekos graph

    Returns
    -------
    igraph.Graph
        An igraph graph with the same structure

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_tsv("graph.tsv")
    >>> G = nekos.to_igraph(g)
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("igraph is required for this function. Install it with: pip install igraph")

    # Get number of nodes and edges
    n = g.num_nodes()
    edges = list(g.get_edges())

    # Create igraph graph
    G = ig.Graph(n, edges, directed=False)

    # Set original node IDs as 'name' attribute
    original_ids = list(g.get_original_ids())
    G.vs['name'] = original_ids

    return G
