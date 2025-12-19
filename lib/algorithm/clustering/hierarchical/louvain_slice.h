#ifndef LOUVAIN_SLICE_H
#define LOUVAIN_SLICE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include "data_structures/graph.h"
#include "data_structures/hierarchical_clustering.h"
#include "algorithm/clustering/hierarchical/leiden_slice.h"  // Reuse Partition structure

// Calculate modularity contribution of a node to a community
inline double calculate_node_modularity_contribution(
    uint32_t node,
    uint32_t community,
    const Graph& g,
    const Partition& partition,
    double resolution,
    double total_edges
) {
    const auto& comm_members = partition.community_members.at(community);

    // Count edges from node to community
    uint32_t edges_to_community = 0;
    for (uint32_t neighbor : g.get_neighbors(node)) {
        if (comm_members.count(neighbor) && neighbor != node) {
            edges_to_community++;
        }
    }

    // Get node degree
    uint32_t node_degree = g.get_neighbors(node).size();

    // Sum degrees in community
    uint32_t comm_degree = 0;
    for (uint32_t member : comm_members) {
        comm_degree += g.get_neighbors(member).size();
    }

    // Modularity contribution = edges_to_comm - resolution * (node_degree * comm_degree) / (2 * total_edges)
    double expected = resolution * (node_degree * comm_degree) / (2.0 * total_edges);
    return static_cast<double>(edges_to_community) - expected;
}

// Local move optimization using modularity
bool local_move_modularity_optimization(
    const Graph& g,
    Partition& partition,
    double resolution,
    std::mt19937& rng,
    bool verbose = false
) {
    bool improved = false;
    uint32_t moves_made = 0;
    double total_edges = g.num_edges;

    // Pre-allocate nodes vector
    std::vector<uint32_t> nodes;
    nodes.reserve(partition.node_to_community.size());
    for (const auto& [node, comm] : partition.node_to_community) {
        nodes.push_back(node);
    }
    std::shuffle(nodes.begin(), nodes.end(), rng);

    // Pre-allocate reusable data structures
    std::unordered_map<uint32_t, uint32_t> comm_edges;

    // Try to move each node
    for (uint32_t node : nodes) {
        uint32_t current_comm = partition.node_to_community[node];
        uint32_t node_degree = g.get_neighbors(node).size();

        // Count edges to each neighboring community
        comm_edges.clear();
        uint32_t edges_to_current = 0;

        for (uint32_t neighbor : g.get_neighbors(node)) {
            auto it = partition.node_to_community.find(neighbor);
            if (it != partition.node_to_community.end() && neighbor != node) {
                uint32_t neighbor_comm = it->second;
                comm_edges[neighbor_comm]++;
                if (neighbor_comm == current_comm) {
                    edges_to_current++;
                }
            }
        }

        // Calculate current community degree (excluding the node)
        uint32_t current_comm_degree = 0;
        for (uint32_t member : partition.community_members[current_comm]) {
            current_comm_degree += g.get_neighbors(member).size();
        }
        current_comm_degree -= node_degree;  // Exclude the node itself

        // Calculate current contribution
        double current_contribution = static_cast<double>(edges_to_current) -
                                     resolution * (node_degree * current_comm_degree) / (2.0 * total_edges);

        // Find best move
        uint32_t best_comm = current_comm;
        double best_delta = 0.0;

        for (const auto& [target_comm, edges_to_target] : comm_edges) {
            if (target_comm == current_comm) continue;

            // Calculate target community degree
            uint32_t target_comm_degree = 0;
            for (uint32_t member : partition.community_members[target_comm]) {
                target_comm_degree += g.get_neighbors(member).size();
            }

            // Calculate contribution if moved to target
            double new_contribution = static_cast<double>(edges_to_target) -
                                     resolution * (node_degree * target_comm_degree) / (2.0 * total_edges);
            double delta = new_contribution - current_contribution;

            if (delta > best_delta + 1e-9) {
                best_delta = delta;
                best_comm = target_comm;
            }
        }

        // Make the move if beneficial
        if (best_comm != current_comm) {
            partition.move_node(node, current_comm, best_comm);
            moves_made++;
            improved = true;
        }
    }

    if (verbose && moves_made > 0) {
        std::cout << "  Local moves: " << moves_made << " nodes moved" << std::endl;
    }

    return improved;
}

// Louvain refinement from dendrogram (uses modularity instead of CPM)
Partition louvain_from_dendrogram(
    const Graph& g,
    const std::vector<DendrogramNode>& dendrogram,
    double resolution,
    uint32_t max_iterations = 10,
    uint32_t random_seed = 42,
    bool verbose = false
) {
    if (verbose) {
        std::cout << "Running Louvain refinement with resolution = " << resolution << std::endl;
    }

    // Calculate distance from resolution (Paris uses distance proportional to 1/resolution)
    double distance = 1.0 / resolution;

    // Extract initial partition from dendrogram
    Partition partition = extract_partition_from_dendrogram(
        dendrogram, g.num_nodes, distance
    );

    if (verbose) {
        std::cout << "Initial partition: " << partition.num_communities()
                  << " communities" << std::endl;
    }

    // Random number generator
    std::mt19937 rng(random_seed);

    // Iterate local moves with refinement
    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        if (verbose) {
            std::cout << "Iteration " << (iter + 1) << "/" << max_iterations << std::endl;
        }

        bool improved = local_move_modularity_optimization(g, partition, resolution, rng, verbose);

        // Refinement phase: split disconnected communities
        refine_partition(g, partition, verbose);

        if (!improved) {
            if (verbose) {
                std::cout << "Converged after " << (iter + 1) << " iterations" << std::endl;
            }
            break;
        }
    }

    if (verbose) {
        std::cout << "Final partition: " << partition.num_communities()
                  << " communities" << std::endl;
    }

    return partition;
}

// Calculate modularity for a partition
double calculate_partition_modularity(
    const Graph& g,
    const Partition& partition,
    double resolution
) {
    double modularity = 0.0;
    double total_edges = g.num_edges;

    for (const auto& [comm_id, members] : partition.community_members) {
        // Count internal edges
        uint32_t internal_edges = 0;
        for (uint32_t node : members) {
            for (uint32_t neighbor : g.get_neighbors(node)) {
                if (members.count(neighbor)) {
                    internal_edges++;
                }
            }
        }
        internal_edges /= 2;  // Each edge counted twice

        // Sum of degrees in community
        uint32_t comm_degree = 0;
        for (uint32_t node : members) {
            comm_degree += g.get_neighbors(node).size();
        }

        // Modularity contribution
        double expected = resolution * (comm_degree * comm_degree) / (4.0 * total_edges);
        modularity += internal_edges - expected;
    }

    // Normalize by total edges
    return modularity / total_edges;
}

#endif // LOUVAIN_SLICE_H
