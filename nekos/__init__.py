"""
nekos - Super-lean library for large-scale graph data
"""

# Import the C++ extension (_core module)
from . import _core

# Export C++ module components
Graph = _core.Graph
from_tsv = _core.from_tsv
has_openmp = _core.has_openmp
Clustering = _core.Clustering
load_clustering_from_tsv = _core.load_clustering_from_tsv

# Import conversion utilities
from .convert import (
    from_networkx,
    to_networkx,
    from_networkit,
    to_networkit,
    from_igraph,
    to_igraph
)

__version__ = "0.1.0"

__all__ = [
    'Graph',
    'from_tsv',
    'has_openmp',
    'Clustering',
    'load_clustering_from_tsv',
    'from_networkx',
    'to_networkx',
    'from_networkit',
    'to_networkit',
    'from_igraph',
    'to_igraph',
]
