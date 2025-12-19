#ifndef HIERARCHICAL_CLUSTERING_H
#define HIERARCHICAL_CLUSTERING_H

#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "graph.h"

// Dendrogram entry: [cluster_a, cluster_b, distance, size]
struct DendrogramNode {
    uint32_t cluster_a;
    uint32_t cluster_b;
    double distance;
    uint32_t size;
};

// Optimized weighted graph structure using vectors for better cache locality
struct WeightedGraph {
    // Map from node ID to adjacency list (neighbor -> weight)
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> adjacency;

    // Use vectors for dense node properties (better cache locality)
    std::vector<double> node_weights;
    std::vector<uint32_t> cluster_sizes;

    // Active nodes set for quick existence checks
    std::unordered_map<uint32_t, bool> active_nodes;

    double total_weight = 0.0;
    uint32_t max_node_id = 0;

    void ensure_capacity(uint32_t node_id) {
        if (node_id >= max_node_id) {
            max_node_id = node_id + 1;
            if (node_weights.size() <= node_id) {
                node_weights.resize(node_id + 1024, 0.0);
                cluster_sizes.resize(node_id + 1024, 0);
            }
        }
    }

    void add_node(uint32_t node) {
        ensure_capacity(node);
        if (active_nodes.find(node) == active_nodes.end()) {
            adjacency[node] = std::unordered_map<uint32_t, double>();
            active_nodes[node] = true;
            if (cluster_sizes[node] == 0) {
                cluster_sizes[node] = 1;
            }
        }
    }

    void add_edge(uint32_t u, uint32_t v, double weight) {
        adjacency[u][v] = weight;
    }

    inline bool has_edge(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return false;
        return it->second.find(v) != it->second.end();
    }

    inline double get_edge_weight(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return 0.0;
        auto it2 = it->second.find(v);
        if (it2 == it->second.end()) return 0.0;
        return it2->second;
    }

    // Return const reference to avoid allocation
    inline const std::unordered_map<uint32_t, double>& get_neighbors_map(uint32_t node) const {
        static const std::unordered_map<uint32_t, double> empty;
        auto it = adjacency.find(node);
        if (it != adjacency.end()) {
            return it->second;
        }
        return empty;
    }

    void get_active_nodes(std::vector<uint32_t>& nodes) const {
        nodes.clear();
        nodes.reserve(active_nodes.size());
        for (const auto& [node, _] : active_nodes) {
            nodes.push_back(node);
        }
    }

    void remove_node(uint32_t node) {
        // Get neighbors before erasing
        auto it = adjacency.find(node);
        if (it != adjacency.end()) {
            // Remove back-edges efficiently
            for (const auto& [neighbor, _] : it->second) {
                auto neighbor_it = adjacency.find(neighbor);
                if (neighbor_it != adjacency.end()) {
                    neighbor_it->second.erase(node);
                }
            }
            adjacency.erase(it);
        }

        active_nodes.erase(node);
        // Keep vectors allocated but mark as inactive
        node_weights[node] = 0.0;
    }

    inline size_t num_nodes() const {
        return active_nodes.size();
    }
};

// Optimized single-pass conversion
inline WeightedGraph convert_to_weighted_graph(const Graph& g) {
    WeightedGraph wg;

    // Pre-allocate vectors
    wg.node_weights.resize(g.num_nodes, 0.0);
    wg.cluster_sizes.resize(g.num_nodes, 1);
    wg.max_node_id = g.num_nodes;
    wg.total_weight = 0.0;

    // Reserve space for adjacency map
    wg.adjacency.reserve(g.num_nodes);
    wg.active_nodes.reserve(g.num_nodes);

    // Single pass: add nodes, edges, and calculate weights
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        wg.adjacency[u] = std::unordered_map<uint32_t, double>();
        wg.active_nodes[u] = true;

        uint32_t degree = g.row_ptr[u + 1] - g.row_ptr[u];
        if (degree > 0) {
            wg.adjacency[u].reserve(degree);
        }

        double node_weight = 0.0;
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            wg.adjacency[u][v] = 1.0;
            node_weight += 1.0;
        }

        wg.node_weights[u] = node_weight;
        wg.total_weight += node_weight;
    }

    return wg;
}

// Reorder dendrogram
inline std::vector<DendrogramNode> reorder_dendrogram(const std::vector<DendrogramNode>& D) {
    if (D.empty()) return D;

    size_t n = D.size() + 1;

    // Create ordering based on distance
    std::vector<std::pair<double, size_t>> order;
    order.reserve(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        order.emplace_back(D[i].distance, i);
    }

    // Sort by distance
    std::sort(order.begin(), order.end());

    // Create index mapping using vector for original nodes (faster than unordered_map)
    std::vector<uint32_t> node_index(n + n - 1);
    for (uint32_t i = 0; i < n; ++i) {
        node_index[i] = i;
    }

    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        node_index[n + orig_idx] = n + t;
    }

    // Reorder dendrogram
    std::vector<DendrogramNode> reordered;
    reordered.reserve(n - 1);
    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        reordered.push_back({
            node_index[D[orig_idx].cluster_a],
            node_index[D[orig_idx].cluster_b],
            D[orig_idx].distance,
            D[orig_idx].size
        });
    }

    return reordered;
}

#endif // HIERARCHICAL_CLUSTERING_H
