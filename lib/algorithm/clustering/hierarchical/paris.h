#ifndef PARIS_H
#define PARIS_H

#include <vector>
#include <limits>
#include <iostream>
#include "../data_structures/hierarchical_clustering.h"
#include "../data_structures/graph.h"

// PARIS algorithm - uses node degree weights in distance metric
std::vector<DendrogramNode> paris(const Graph& g, bool verbose = false) {
    // Convert to weighted graph
    WeightedGraph F = convert_to_weighted_graph(g);
    uint32_t n = F.num_nodes();

    if (verbose) {
        std::cout << "Running PARIS algorithm on graph with " << n << " nodes" << std::endl;
        std::cout << "Total weight: " << F.total_weight << std::endl;
    }

    // Pre-allocate dendrogram
    std::vector<DendrogramNode> dendrogram;
    dendrogram.reserve(n - 1);

    // Connected components
    std::vector<std::pair<uint32_t, uint32_t>> connected_components;

    // Cluster index
    uint32_t cluster_idx = n;

    // Reusable vectors to avoid repeated allocations
    std::vector<uint32_t> active_nodes;
    std::vector<uint32_t> chain;
    chain.reserve(1024);

    while (F.num_nodes() > 0) {
        // Start nearest-neighbor chain
        chain.clear();
        F.get_active_nodes(active_nodes);
        if (active_nodes.empty()) break;

        chain.push_back(active_nodes[0]);

        while (!chain.empty()) {
            uint32_t a = chain.back();
            chain.pop_back();

            // Find nearest neighbor - PARIS distance metric
            double dmin = std::numeric_limits<double>::infinity();
            int32_t b = -1;

            const auto& neighbors = F.get_neighbors_map(a);
            double node_weight_a = F.node_weights[a];
            double inv_total_weight = 1.0 / F.total_weight;

            for (const auto& [v, edge_weight] : neighbors) {
                if (v != a) {
                    // PARIS distance: d = p(i) * p(j) / p(i,j) / total_weight
                    double d = F.node_weights[v] * node_weight_a / edge_weight * inv_total_weight;

                    if (d < dmin) {
                        b = v;
                        dmin = d;
                    } else if (d == dmin && static_cast<int32_t>(v) < b) {
                        b = v;
                    }
                }
            }

            double d = dmin;

            if (!chain.empty()) {
                uint32_t c = chain.back();
                chain.pop_back();

                if (b == static_cast<int32_t>(c)) {
                    // Merge a and b
                    uint32_t size_a = F.cluster_sizes[a];
                    uint32_t size_b = F.cluster_sizes[b];

                    dendrogram.push_back({a, static_cast<uint32_t>(b), d, size_a + size_b});

                    if (verbose && dendrogram.size() % 1000 == 0) {
                        std::cout << "Merged " << dendrogram.size() << " clusters" << std::endl;
                    }

                    // Update graph - add new cluster node
                    F.add_node(cluster_idx);

                    // Copy neighbor lists before modification (references would be invalidated)
                    auto neighbors_a_copy = F.adjacency[a];
                    auto neighbors_b_copy = F.adjacency[b];

                    // Reserve space for new adjacency list
                    size_t total_neighbors = neighbors_a_copy.size() + neighbors_b_copy.size();
                    F.adjacency[cluster_idx].reserve(total_neighbors);

                    // Add edges from new cluster to neighbors of a
                    for (const auto& [v, weight] : neighbors_a_copy) {
                        F.add_edge(cluster_idx, v, weight);
                        F.add_edge(v, cluster_idx, weight);
                    }

                    // Add edges from new cluster to neighbors of b
                    for (const auto& [v, weight_b] : neighbors_b_copy) {
                        if (F.has_edge(cluster_idx, v)) {
                            double existing_weight = F.get_edge_weight(cluster_idx, v);
                            double new_weight = existing_weight + weight_b;
                            F.add_edge(cluster_idx, v, new_weight);
                            F.add_edge(v, cluster_idx, new_weight);
                        } else {
                            F.add_edge(cluster_idx, v, weight_b);
                            F.add_edge(v, cluster_idx, weight_b);
                        }
                    }

                    // Update node weight and size
                    F.node_weights[cluster_idx] = F.node_weights[a] + F.node_weights[b];
                    F.cluster_sizes[cluster_idx] = size_a + size_b;

                    // Remove old nodes
                    F.remove_node(a);
                    F.remove_node(b);

                    // Increment cluster index
                    cluster_idx++;
                } else {
                    chain.push_back(c);
                    chain.push_back(a);
                    chain.push_back(b);
                }
            } else if (b >= 0) {
                chain.push_back(a);
                chain.push_back(b);
            } else {
                // Isolated node - add to connected components
                connected_components.push_back({a, F.cluster_sizes[a]});
                F.remove_node(a);
            }
        }
    }

    // Add connected components to dendrogram
    if (!connected_components.empty()) {
        auto [a, size_a] = connected_components.back();
        connected_components.pop_back();

        uint32_t total_size = size_a;
        for (const auto& [b, size_b] : connected_components) {
            total_size += size_b;
            dendrogram.push_back({a, b, std::numeric_limits<double>::infinity(), total_size});
            a = cluster_idx;
            cluster_idx++;
        }
    }

    if (verbose) {
        std::cout << "Created dendrogram with " << dendrogram.size() << " merges" << std::endl;
    }

    // Reorder dendrogram
    return reorder_dendrogram(dendrogram);
}

#endif // PARIS_H
