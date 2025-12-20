"""
nekos - Super-lean library for large-scale graph data
"""

# Import the C++ extension (_core module)
from . import _core

# Export C++ module components
Graph = _core.Graph
from_csv = _core.from_csv
has_openmp = _core.has_openmp
Clustering = _core.Clustering
HierarchicalClustering = _core.HierarchicalClustering
load_clustering_from_csv = _core.load_clustering_from_csv
find_connected_components = _core.find_connected_components

# Backward compatibility aliases
from_tsv = lambda filename, num_threads=1, verbose=False, skip_header=False: \
    from_csv(filename, num_threads=num_threads, verbose=verbose,
             header=0 if skip_header else None, delimiter='\t')

load_clustering_from_tsv = lambda filename, graph, verbose=False, skip_header=False: \
    load_clustering_from_csv(filename, graph, verbose=verbose,
                             header=0 if skip_header else None, delimiter='\t')

# Import conversion utilities
from .convert import (
    from_networkx,
    to_networkx,
    from_networkit,
    to_networkit,
    from_igraph,
    to_igraph
)

# Import clustering module
from . import clustering

__version__ = "0.1.0"

__all__ = [
    'Graph',
    'from_csv',
    'from_tsv',  # Backward compatibility
    'has_openmp',
    'Clustering',
    'HierarchicalClustering',
    'load_clustering_from_csv',
    'load_clustering_from_tsv',  # Backward compatibility
    'find_connected_components',
    'from_networkx',
    'to_networkx',
    'from_networkit',
    'to_networkit',
    'from_igraph',
    'to_igraph',
    'clustering',
]
