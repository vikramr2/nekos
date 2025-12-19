"""
Hierarchical clustering algorithms (Paris and Champaign)
"""

from .._core import champaign as _champaign
from .._core import paris as _paris
from .._core import load_hierarchical_from_json as _load_hierarchical_from_json

def champaign(graph, verbose=False):
    """
    Run Champaign hierarchical clustering algorithm.

    Champaign uses a size-based distance metric:
    d = (n_a * n_b) / (total_weight * p(a,b))

    where n_a and n_b are cluster sizes.

    Parameters
    ----------
    graph : nekos.Graph
        Input graph to cluster
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    HierarchicalClustering
        Hierarchical clustering dendrogram object with methods:
        - leiden(gamma): Extract Leiden clustering at resolution gamma
        - louvain(resolution): Extract Louvain clustering at resolution
        - save_json(filename): Save dendrogram to JSON file

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_csv('graph.csv')
    >>> dendro = nekos.clustering.hierarchical.champaign(g)
    >>> clustering = dendro.leiden(gamma=0.5)
    """
    return _champaign(graph, verbose=verbose)

def paris(graph, verbose=False):
    """
    Run Paris hierarchical clustering algorithm.

    Paris uses a degree-based distance metric:
    d = p(i) * p(j) / p(i,j) / total_weight

    where p(i) and p(j) are node degrees.

    Parameters
    ----------
    graph : nekos.Graph
        Input graph to cluster
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    HierarchicalClustering
        Hierarchical clustering dendrogram object with methods:
        - leiden(gamma): Extract Leiden clustering at resolution gamma
        - louvain(resolution): Extract Louvain clustering at resolution
        - save_json(filename): Save dendrogram to JSON file

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_csv('graph.csv')
    >>> dendro = nekos.clustering.hierarchical.paris(g)
    >>> clustering = dendro.louvain(resolution=0.5)
    """
    return _paris(graph, verbose=verbose)

def load_from_json(filename, graph):
    """
    Load hierarchical clustering from JSON file.

    Parameters
    ----------
    filename : str
        Path to JSON file containing dendrogram
    graph : nekos.Graph
        Graph object (needed for computing slices)

    Returns
    -------
    HierarchicalClustering
        Hierarchical clustering dendrogram object

    Examples
    --------
    >>> import nekos
    >>> g = nekos.from_csv('graph.csv')
    >>> dendro = nekos.clustering.hierarchical.load_from_json('dendro.json', g)
    >>> clustering = dendro.leiden(gamma=0.5)
    """
    return _load_hierarchical_from_json(filename, graph)

__all__ = ['champaign', 'paris', 'load_from_json']
