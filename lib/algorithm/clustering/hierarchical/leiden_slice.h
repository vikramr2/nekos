#ifndef LEIDEN_SLICE_H
#define LEIDEN_SLICE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include "../data_structures/graph.h"
#include "../data_structures/hierarchical_clustering.h"

// Partition representation
struct Partition {
    std::unordered_map<uint32_t, uint32_t> node_to_community;
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> community_members;

    void add_node_to_community(uint32_t node, uint32_t comm) {
        node_to_community[node] = comm;
        community_members[comm].insert(node);
    }

    void remove_node_from_community(uint32_t node, uint32_t comm) {
        community_members[comm].erase(node);
        if (community_members[comm].empty()) {
            community_members.erase(comm);
        }
    }

    void move_node(uint32_t node, uint32_t from_comm, uint32_t to_comm) {
        remove_node_from_community(node, from_comm);
        add_node_to_community(node, to_comm);
    }

    uint32_t num_communities() const {
        return community_members.size();
    }
};

// Calculate CPM contribution of a node to a community
inline double calculate_node_cpm_contribution(
    uint32_t node,
    uint32_t community,
    const Graph& g,
    const Partition& partition,
    double gamma
) {
    const auto& comm_members = partition.community_members.at(community);

    // Count edges from node to community
    uint32_t edges_to_community = 0;
    for (uint32_t neighbor : g.get_neighbors(node)) {
        if (comm_members.count(neighbor) && neighbor != node) {
            edges_to_community++;
        }
    }

    // CPM contribution = edges - gamma * (size - 1)
    // (size - 1) because we don't count the node pairing with itself
    uint32_t comm_size = comm_members.size();
    return static_cast<double>(edges_to_community) - gamma * (comm_size - 1);
}

// Local move optimization (Leiden phase 1) - Optimized version
bool local_move_optimization(
    const Graph& g,
    Partition& partition,
    double gamma,
    std::mt19937& rng,
    bool verbose = false
) {
    bool improved = false;
    uint32_t moves_made = 0;

    // Pre-allocate nodes vector (reuse across iterations)
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
        uint32_t current_comm_size = partition.community_members[current_comm].size();

        // Count edges to each neighboring community (single pass)
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

        // Calculate current contribution (without function call overhead)
        double current_contribution = static_cast<double>(edges_to_current) -
                                     gamma * (current_comm_size - 1);

        // Find best move
        uint32_t best_comm = current_comm;
        double best_delta = 0.0;

        for (const auto& [target_comm, edges_to_target] : comm_edges) {
            if (target_comm == current_comm) continue;

            uint32_t target_comm_size = partition.community_members[target_comm].size();

            // Calculate contribution if moved to target
            double new_contribution = static_cast<double>(edges_to_target) -
                                     gamma * target_comm_size;  // +1 for node, -1 for not counting self
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

// Extract partition from dendrogram at given distance
Partition extract_partition_from_dendrogram(
    const std::vector<DendrogramNode>& dendrogram,
    uint32_t num_nodes,
    double max_distance
) {
    Partition partition;

    // Initialize: each node is its own community
    for (uint32_t i = 0; i < num_nodes; i++) {
        partition.add_node_to_community(i, i);
    }

    // Apply merges up to max_distance
    uint32_t cluster_id = num_nodes;
    std::unordered_map<uint32_t, uint32_t> cluster_to_community;

    for (uint32_t i = 0; i < num_nodes; i++) {
        cluster_to_community[i] = i;
    }

    for (const auto& merge : dendrogram) {
        if (merge.distance > max_distance ||
            merge.distance == std::numeric_limits<double>::infinity()) {
            break;
        }

        // Find communities of the two clusters being merged
        uint32_t comm_a = cluster_to_community[merge.cluster_a];
        uint32_t comm_b = cluster_to_community[merge.cluster_b];

        if (comm_a == comm_b) {
            cluster_to_community[cluster_id] = comm_a;
        } else {
            // Merge: move all nodes from comm_b to comm_a
            auto members_b = partition.community_members[comm_b];
            for (uint32_t node : members_b) {
                partition.move_node(node, comm_b, comm_a);
            }
            cluster_to_community[cluster_id] = comm_a;
        }

        cluster_id++;
    }

    return partition;
}

// Find connected components within a subset of nodes using BFS
std::vector<std::unordered_set<uint32_t>> find_connected_components(
    const Graph& g,
    const std::unordered_set<uint32_t>& nodes
) {
    std::vector<std::unordered_set<uint32_t>> components;
    std::unordered_set<uint32_t> visited;

    for (uint32_t start_node : nodes) {
        if (visited.count(start_node)) continue;

        // BFS to find connected component
        std::unordered_set<uint32_t> component;
        std::vector<uint32_t> queue;
        queue.push_back(start_node);
        visited.insert(start_node);
        component.insert(start_node);

        size_t queue_idx = 0;
        while (queue_idx < queue.size()) {
            uint32_t current = queue[queue_idx++];

            // Check all neighbors
            for (uint32_t neighbor : g.get_neighbors(current)) {
                // Only consider neighbors within the node subset
                if (nodes.count(neighbor) && !visited.count(neighbor)) {
                    visited.insert(neighbor);
                    component.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

// Refinement phase: splits disconnected communities into separate components
void refine_partition(
    const Graph& g,
    Partition& partition,
    bool verbose = false
) {
    uint32_t communities_split = 0;
    uint32_t next_comm_id = 0;

    // Find next available community ID
    for (const auto& [comm_id, members] : partition.community_members) {
        if (comm_id >= next_comm_id) {
            next_comm_id = comm_id + 1;
        }
    }

    // Make a copy of community IDs to iterate over (since we'll modify the map)
    std::vector<uint32_t> comm_ids;
    for (const auto& [comm_id, members] : partition.community_members) {
        comm_ids.push_back(comm_id);
    }

    for (uint32_t comm_id : comm_ids) {
        // Check if community still exists (might have been split already)
        if (partition.community_members.find(comm_id) == partition.community_members.end()) {
            continue;
        }

        const auto& members = partition.community_members[comm_id];

        // Skip singleton communities
        if (members.size() <= 1) continue;

        // Find connected components within this community
        auto components = find_connected_components(g, members);

        // If more than one component, split the community
        if (components.size() > 1) {
            communities_split++;

            // Keep the first component in the original community
            // Move other components to new communities
            bool first = true;
            for (const auto& component : components) {
                if (first) {
                    first = false;
                    // Update the original community to only contain this component
                    partition.community_members[comm_id] = component;
                    for (uint32_t node : component) {
                        partition.node_to_community[node] = comm_id;
                    }
                } else {
                    // Create new community for this component
                    uint32_t new_comm_id = next_comm_id++;
                    for (uint32_t node : component) {
                        partition.node_to_community[node] = new_comm_id;
                        partition.community_members[new_comm_id].insert(node);
                    }
                }
            }
        }
    }

    if (verbose && communities_split > 0) {
        std::cout << "  Refinement: split " << communities_split
                  << " disconnected communities" << std::endl;
    }
}

// Leiden refinement from dendrogram
Partition leiden_from_dendrogram(
    const Graph& g,
    const std::vector<DendrogramNode>& dendrogram,
    double gamma,
    uint32_t max_iterations = 10,
    uint32_t random_seed = 42,
    bool verbose = false
) {
    if (verbose) {
        std::cout << "Running Leiden refinement with gamma = " << gamma << std::endl;
    }

    // Calculate distance from gamma
    double distance = 1.0 / gamma;

    // Extract initial partition
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

        bool improved = local_move_optimization(g, partition, gamma, rng, verbose);

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

// Calculate CPM for a partition
double calculate_partition_cpm(
    const Graph& g,
    const Partition& partition,
    double gamma
) {
    double cpm = 0.0;

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

        // CPM contribution
        uint32_t n = members.size();
        double penalty = gamma * n * (n - 1) / 2.0;
        cpm += internal_edges - penalty;
    }

    return cpm;
}

#endif // LEIDEN_SLICE_H
