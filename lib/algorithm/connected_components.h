#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include "../data_structures/graph.h"
#include "../data_structures/clustering.h"

// Find connected components using BFS
std::vector<std::vector<uint32_t>> find_connected_components(const Graph& graph) {
    std::vector<std::vector<uint32_t>> components;

    if (graph.num_nodes == 0) return components;

    std::vector<bool> visited(graph.num_nodes, false);

    for (uint32_t start = 0; start < graph.num_nodes; start++) {
        if (visited[start]) continue;

        // BFS to find component
        std::vector<uint32_t> component;
        std::queue<uint32_t> queue;

        queue.push(start);
        visited[start] = true;

        while (!queue.empty()) {
            uint32_t node = queue.front();
            queue.pop();
            component.push_back(node);

            std::vector<uint32_t> neighbors = graph.get_neighbors(node);
            for (uint32_t neighbor : neighbors) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

// Get component sizes
std::unordered_map<size_t, size_t> get_component_sizes(const std::vector<std::vector<uint32_t>>& components) {
    std::unordered_map<size_t, size_t> sizes;
    for (size_t i = 0; i < components.size(); i++) {
        sizes[i] = components[i].size();
    }
    return sizes;
}

// Convert connected components to a Clustering object
Clustering components_to_clustering(const std::vector<std::vector<uint32_t>>& components) {
    Clustering clustering;

    // Find the maximum node ID to properly size the clustering
    uint32_t max_node_id = 0;
    for (const auto& component : components) {
        for (uint32_t node_id : component) {
            if (node_id > max_node_id) {
                max_node_id = node_id;
            }
        }
    }

    // Initialize clustering with enough space for all nodes
    clustering.reset(max_node_id + 1);

    // Assign each component to a cluster with ID "component_<index>"
    for (size_t i = 0; i < components.size(); i++) {
        std::string cluster_id = "component_" + std::to_string(i);
        for (uint32_t node_id : components[i]) {
            clustering.assign_node_to_cluster(node_id, cluster_id);
        }
    }

    return clustering;
}

#endif // CONNECTED_COMPONENTS_H
