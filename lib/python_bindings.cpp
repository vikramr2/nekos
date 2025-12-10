#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "data_structures/graph.h"
#include "io/graph_io.h"
#include "algorithm/layout.h"

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

    // Compute graph layout (returns list of (x, y) tuples)
    py::list compute_layout(const std::string& layout = "force", int iterations = 50,
                           int num_threads = 1, unsigned int seed = 42,
                           bool verbose = false) const {
        std::vector<Point2D> positions;

        if (layout == "force") {
            positions = force_directed_layout(g, iterations, -1.0f, num_threads, seed, verbose);
        } else if (layout == "random") {
            positions = random_layout(g, seed);
        } else {
            throw std::invalid_argument("Unknown layout: " + layout);
        }

        // Convert to Python list of tuples
        py::list result;
        for (const auto& pos : positions) {
            result.append(py::make_tuple(pos.x, pos.y));
        }
        return result;
    }

    // Visualize graph (calls Python visualize module)
    void visualize(const std::string& layout = "force", int iterations = 50,
                   float node_size = 5.0, float edge_width = 1.0,
                   py::object node_color = py::make_tuple(1.0f, 1.0f, 1.0f, 1.0f),
                   py::object edge_color = py::make_tuple(0.8f, 0.8f, 0.8f, 1.0f),
                   py::object background_color = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
                   int num_threads = 1, bool verbose = false) const {
        py::module_ visualize_module = py::module_::import("nekos.visualize");
        py::object visualize_func = visualize_module.attr("visualize");

        // Convert this GraphWrapper to a Python object and call visualize
        py::object graph_obj = py::cast(this);
        visualize_func(graph_obj, py::arg("layout") = layout,
                      py::arg("iterations") = iterations,
                      py::arg("node_size") = node_size,
                      py::arg("edge_width") = edge_width,
                      py::arg("node_color") = node_color,
                      py::arg("edge_color") = edge_color,
                      py::arg("background_color") = background_color,
                      py::arg("num_threads") = num_threads,
                      py::arg("verbose") = verbose);
    }
};

// Load graph from TSV
GraphWrapper* load_from_tsv(const std::string& filename, int num_threads = 1, bool verbose = false) {
    auto* wrapper = new GraphWrapper();
    wrapper->g = load_undirected_tsv_edgelist_parallel(filename, num_threads, verbose);
    return wrapper;
}

// Pybind11 module definition
PYBIND11_MODULE(_core, m) {
    m.doc() = "Graph data structures and I/O utilities (C++ core)";

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
        .def("compute_layout", &GraphWrapper::compute_layout,
             "Compute graph layout and return node positions as list of (x, y) tuples",
             py::arg("layout") = "force",
             py::arg("iterations") = 50,
             py::arg("num_threads") = 1,
             py::arg("seed") = 42,
             py::arg("verbose") = false)
        .def("visualize", &GraphWrapper::visualize,
             "Visualize graph using vispy (OpenGL-accelerated)",
             py::arg("layout") = "force",
             py::arg("iterations") = 50,
             py::arg("node_size") = 5.0,
             py::arg("edge_width") = 1.0,
             py::arg("node_color") = py::make_tuple(1.0f, 1.0f, 1.0f, 1.0f),
             py::arg("edge_color") = py::make_tuple(0.8f, 0.8f, 0.8f, 1.0f),
             py::arg("background_color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
             py::arg("num_threads") = 1,
             py::arg("verbose") = false)
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
