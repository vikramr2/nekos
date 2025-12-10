#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "data_structures/graph.h"
#include "io/graph_io.h"

namespace py = pybind11;

// Python-friendly graph wrapper
class GraphWrapper {
public:
    Graph g;

    GraphWrapper() : g() {}

    void add_edge(uint32_t u, uint32_t v) {
        g.add_edge(u, v);
    }

    uint32_t add_node(uint64_t original_id) {
        return g.add_node(original_id);
    }

    uint32_t num_nodes() const {
        return g.num_nodes;
    }

    uint32_t num_edges() const {
        return g.num_edges;
    }

    uint32_t get_degree(uint32_t node) const {
        return g.get_degree(node);
    }

    std::vector<uint32_t> neighbors(uint32_t node) const {
        return g.get_neighbors(node);
    }

    // Get original node ID from internal ID
    uint64_t get_original_id(uint32_t internal_id) const {
        if (internal_id >= g.id_map.size()) {
            throw std::out_of_range("Internal ID out of range");
        }
        return g.id_map[internal_id];
    }

    // Get internal ID from original node ID
    uint32_t get_internal_id(uint64_t original_id) const {
        auto it = g.node_map.find(original_id);
        if (it == g.node_map.end()) {
            throw std::out_of_range("Original ID not found in graph");
        }
        return it->second;
    }

    // Get all original node IDs
    py::list get_original_ids() const {
        py::list result;
        for (uint64_t original_id : g.id_map) {
            result.append(original_id);
        }
        return result;
    }

    // Get all edges as list of tuples
    py::list get_edges() const {
        py::list result;
        std::vector<std::tuple<uint32_t, uint32_t>> edges = ::get_edges(g);
        for (const auto& edge : edges) {
            result.append(py::make_tuple(std::get<0>(edge), std::get<1>(edge)));
        }
        return result;
    }

    // Save graph to TSV file
    void save_tsv(const std::string& filename, bool verbose = false) const {
        save_graph_edgelist(filename, g, verbose);
    }
};

// Load graph from TSV
GraphWrapper* load_from_tsv(const std::string& filename, int num_threads = 1, bool verbose = false) {
    auto* wrapper = new GraphWrapper();
    wrapper->g = load_undirected_tsv_edgelist_parallel(filename, num_threads, verbose);
    return wrapper;
}

// Pybind11 module definition
PYBIND11_MODULE(nekos, m) {
    m.doc() = "Graph data structures and I/O utilities";

    // Check if OpenMP is available and warn if not
    #ifdef NEKOS_HAS_OPENMP
    m.attr("has_openmp") = true;
    #else
    m.attr("has_openmp") = false;
    // Issue warning on module import
    PyErr_WarnEx(PyExc_UserWarning,
                 "OpenMP not found during build. Falling back to serial processing. "
                 "For better performance, install OpenMP and rebuild the package.",
                 1);
    #endif

    // Graph class
    py::class_<GraphWrapper>(m, "Graph")
        .def(py::init<>(), "Create an empty graph")
        .def("add_edge", &GraphWrapper::add_edge,
             "Add an edge to the graph",
             py::arg("u"), py::arg("v"))
        .def("add_node", &GraphWrapper::add_node,
             "Add a node with an original ID and return its internal ID",
             py::arg("original_id"))
        .def("num_nodes", &GraphWrapper::num_nodes,
             "Get number of nodes")
        .def("num_edges", &GraphWrapper::num_edges,
             "Get number of edges")
        .def("get_degree", &GraphWrapper::get_degree,
             "Get degree of a node",
             py::arg("node"))
        .def("neighbors", &GraphWrapper::neighbors,
             "Get neighbors of a node",
             py::arg("node"))
        .def("get_original_id", &GraphWrapper::get_original_id,
             "Get original node ID from internal ID",
             py::arg("internal_id"))
        .def("get_internal_id", &GraphWrapper::get_internal_id,
             "Get internal ID from original node ID",
             py::arg("original_id"))
        .def("get_original_ids", &GraphWrapper::get_original_ids,
             "Get list of all original node IDs")
        .def("get_edges", &GraphWrapper::get_edges,
             "Get all edges as list of (u, v) tuples")
        .def("save_tsv", &GraphWrapper::save_tsv,
             "Save graph to TSV file",
             py::arg("filename"), py::arg("verbose") = false)
        .def("__repr__", [](const GraphWrapper& g) {
            return "Graph(nodes=" + std::to_string(g.num_nodes()) +
                   ", edges=" + std::to_string(g.num_edges()) + ")";
        });

    // Module functions
    m.def("from_tsv", &load_from_tsv,
          "Load graph from TSV file",
          py::arg("filename"),
          py::arg("num_threads") = 1,
          py::arg("verbose") = false,
          py::return_value_policy::take_ownership);
}
