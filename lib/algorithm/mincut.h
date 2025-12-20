#ifndef NEKOS_MINCUT_H
#define NEKOS_MINCUT_H

#include <memory>
#include <vector>
#include <tuple>

// Include our own graph data structure
#include "data_structures/graph.h"
#include "data_structures/mincut_result.h"

// Include VieCut headers from extlib
#include "extlib/VieCut/lib/common/definitions.h"
#include "extlib/VieCut/lib/common/configuration.h"
#include "extlib/VieCut/lib/data_structure/mutable_graph.h"
#include "extlib/VieCut/lib/tools/timer.h"
#include "extlib/VieCut/lib/tools/random_functions.h"
#include "extlib/VieCut/lib/algorithms/global_mincut/algorithms.h"

// Type alias for VieCut graph pointer
using GraphPtr = std::shared_ptr<mutable_graph>;

static std::shared_ptr<mutable_graph> convert_to_viecut(const Graph& g) {
    // Create a new viecut mutable_graph object
    auto G = std::make_shared<mutable_graph>();

    // Initialize the graph with the number of nodes and edges
    std::vector<std::tuple<uint32_t, uint32_t> > edges = get_edges(g);

    size_t nmbNodes = g.num_nodes;
    size_t nmbEdges = g.num_edges;

    NodeID node_counter = 0;
    EdgeID edge_counter = 0;

    G->start_construction(nmbNodes, nmbEdges);

    // Add nodes to the graph
    for (int i = 0; i < nmbNodes; i++) {
        NodeID node = G->new_node();
        node_counter++;
        G->setPartitionIndex(node, 0);
    }

    // Add edges to the graph
    for (std::tuple<int, int> edge : edges) {
        NodeID u = std::get<0>(edge);
        NodeID v = std::get<1>(edge);

        G->new_edge(u, v, 1);
    }

    // Finalize the graph construction
    G->finish_construction();
    G->computeDegrees();

    return G;
}

MincutResult compute_mincut(const Graph& g) {
    // Set the algorithm and queue type
    auto cfg = configuration::getConfig();
    cfg->algorithm = "cactus";
    cfg->queue_type = "bqueue";
    cfg->find_most_balanced_cut = true;
    cfg->save_cut = true;

    std::vector<int> light;
    std::vector<int> heavy;

    timer t;

    // Convert to viecut graph_access
    auto G = convert_to_viecut(g);

    timer tdegs;

    random_functions::setSeed(0);

    NodeID n = G->number_of_nodes();
    EdgeID m = G->number_of_edges();

    auto mc = selectMincutAlgorithm<GraphPtr>("cactus");

    t.restart();
    EdgeWeight cut;
    cut = mc->perform_minimum_cut(G);

    for (NodeID node : G->nodes()) {
        if (G->getNodeInCut(node)) {
            light.push_back(node);
        } else {
            heavy.push_back(node);
        }
    }

    free(mc);

    return MincutResult(light, heavy, cut);
}

#endif // NEKOS_MINCUT_H
