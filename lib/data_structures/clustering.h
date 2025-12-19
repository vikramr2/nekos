#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <iostream>

#include "graph.h"

// Dual representation of clustering information
struct Clustering {
    // Maps node ID to cluster index (internal representation)
    std::vector<uint32_t> node_to_cluster_idx;
    
    // Maps cluster index to the set of nodes in that cluster
    std::vector<std::unordered_set<uint32_t>> cluster_nodes;

    // Maps cluster index to the set of missing nodes in that cluster
    std::vector<std::unordered_set<uint32_t>> cluster_missing_nodes;
    
    // Maps cluster index to original cluster ID (string)
    std::vector<std::string> cluster_ids;
    
    // Maps original cluster ID to internal cluster index
    std::unordered_map<std::string, uint32_t> cluster_id_to_idx;
    
    // Get all nodes in a specific cluster by original ID
    const std::unordered_set<uint32_t>& get_cluster_nodes(const std::string& cluster_id) const {
        static const std::unordered_set<uint32_t> empty_set;
        auto it = cluster_id_to_idx.find(cluster_id);
        if (it == cluster_id_to_idx.end()) {
            return empty_set;
        }
        return cluster_nodes[it->second];
    }

    // Get all missing nodes in a specific cluster by original ID
    const std::unordered_set<uint32_t>& get_cluster_missing_nodes(const std::string& cluster_id) const {
        static const std::unordered_set<uint32_t> empty_set;
        auto it = cluster_id_to_idx.find(cluster_id);
        if (it == cluster_id_to_idx.end()) {
            return empty_set;
        }
        return cluster_missing_nodes[it->second];
    }
    
    // Get all nodes in a specific cluster by internal index
    const std::unordered_set<uint32_t>& get_cluster_nodes_by_idx(uint32_t cluster_idx) const {
        static const std::unordered_set<uint32_t> empty_set;
        if (cluster_idx >= cluster_nodes.size()) {
            return empty_set;
        }
        return cluster_nodes[cluster_idx];
    }
    
    // Get the cluster ID for a specific node
    std::string get_node_cluster(uint32_t node_id) const {
        if (node_id >= node_to_cluster_idx.size() || 
            node_to_cluster_idx[node_id] == UINT32_MAX || 
            node_to_cluster_idx[node_id] >= cluster_ids.size()) {
            return "";
        }
        return cluster_ids[node_to_cluster_idx[node_id]];
    }
    
    // Check if a cluster exists by original ID
    bool has_cluster(const std::string& cluster_id) const {
        auto it = cluster_id_to_idx.find(cluster_id);
        if (it == cluster_id_to_idx.end()) {
            return false;
        }
        return !cluster_nodes[it->second].empty();
    }
    
    // Get internal cluster index from original ID
    uint32_t get_cluster_idx(const std::string& cluster_id) const {
        auto it = cluster_id_to_idx.find(cluster_id);
        if (it == cluster_id_to_idx.end()) {
            return UINT32_MAX;
        }
        return it->second;
    }
    
    // Get original cluster ID from internal index
    const std::string& get_cluster_id(uint32_t cluster_idx) const {
        static const std::string empty_string;
        if (cluster_idx >= cluster_ids.size()) {
            return empty_string;
        }
        return cluster_ids[cluster_idx];
    }
    
    // Add a new cluster or get existing cluster index
    uint32_t add_or_get_cluster(const std::string& cluster_id) {
        auto it = cluster_id_to_idx.find(cluster_id);
        if (it != cluster_id_to_idx.end()) {
            return it->second;
        }
        
        // Add new cluster
        uint32_t new_idx = cluster_ids.size();
        cluster_id_to_idx[cluster_id] = new_idx;
        cluster_ids.push_back(cluster_id);
        cluster_nodes.emplace_back();
        cluster_missing_nodes.emplace_back();
        return new_idx;
    }
    
    // Assign a node to a cluster
    void assign_node_to_cluster(uint32_t node_id, const std::string& cluster_id) {
        // Ensure vector is large enough
        if (node_id >= node_to_cluster_idx.size()) {
            node_to_cluster_idx.resize(node_id + 1, UINT32_MAX);
        }

        // First, if node already has a cluster, remove it from that cluster
        if (node_to_cluster_idx[node_id] != UINT32_MAX) {
            uint32_t old_cluster_idx = node_to_cluster_idx[node_id];
            if (old_cluster_idx < cluster_nodes.size()) {
                cluster_nodes[old_cluster_idx].erase(node_id);
            }
        }

        // Now assign to new cluster
        uint32_t cluster_idx = add_or_get_cluster(cluster_id);
        node_to_cluster_idx[node_id] = cluster_idx;
        cluster_nodes[cluster_idx].insert(node_id);
    }

    // Assign a missing node to a cluster
    void assign_missing_node_to_cluster(uint32_t node_id, const std::string& cluster_id) {
        uint32_t cluster_idx = add_or_get_cluster(cluster_id);
        cluster_missing_nodes[cluster_idx].insert(node_id);
    }
    
    // Reset/initialize the clustering
    void reset(size_t num_nodes) {
        node_to_cluster_idx.resize(num_nodes, UINT32_MAX);  // Initialize with invalid cluster indices
        cluster_nodes.clear();
        cluster_ids.clear();
        cluster_id_to_idx.clear();
    }
    
    // Verify that the clustering is valid (each node belongs to at most one cluster)
    bool verify(bool verbose = false) const {
        bool valid = true;
        
        // Check that each node belongs to at most one valid cluster
        for (uint32_t node_id = 0; node_id < node_to_cluster_idx.size(); ++node_id) {
            uint32_t cluster_idx = node_to_cluster_idx[node_id];
            
            // Skip unclustered nodes
            if (cluster_idx == UINT32_MAX) {
                continue;
            }
            
            // Check that the cluster exists and contains the node
            if (cluster_idx >= cluster_nodes.size()) {
                if (verbose) {
                    std::cout << "Invalid cluster index for node " << node_id << ": " 
                              << cluster_idx << " >= " << cluster_nodes.size() << std::endl;
                }
                valid = false;
                continue;
            }
            
            if (cluster_nodes[cluster_idx].count(node_id) == 0) {
                if (verbose) {
                    std::cout << "Node " << node_id << " is assigned to cluster index " 
                              << cluster_idx << " but not found in that cluster's node set" << std::endl;
                }
                valid = false;
            }
        }
        
        // Check that each cluster's nodes have the correct cluster assignment
        for (uint32_t cluster_idx = 0; cluster_idx < cluster_nodes.size(); ++cluster_idx) {
            for (uint32_t node_id : cluster_nodes[cluster_idx]) {
                if (node_id >= node_to_cluster_idx.size()) {
                    if (verbose) {
                        std::cout << "Invalid node ID in cluster " << cluster_idx << ": " 
                                  << node_id << " >= " << node_to_cluster_idx.size() << std::endl;
                    }
                    valid = false;
                    continue;
                }
                
                if (node_to_cluster_idx[node_id] != cluster_idx) {
                    if (verbose) {
                        std::cout << "Node " << node_id << " is in cluster index " << cluster_idx 
                                  << " but assigned to cluster index " << node_to_cluster_idx[node_id] << std::endl;
                    }
                    valid = false;
                }
            }
        }
        
        return valid;
    }
    
    // Get the number of nodes that have a valid cluster assignment
    size_t get_clustered_node_count() const {
        size_t count = 0;
        for (uint32_t cluster_idx : node_to_cluster_idx) {
            if (cluster_idx != UINT32_MAX) {
                count++;
            }
        }
        return count;
    }
    
    // Get the number of clusters (excluding empty ones)
    size_t get_non_empty_cluster_count() const {
        size_t count = 0;
        for (const auto& nodes : cluster_nodes) {
            if (!nodes.empty()) {
                count++;
            }
        }
        return count;
    }

    // Get the total number of clusters (including empty ones)
    size_t size() const {
        return cluster_nodes.size();
    }
};

#endif // CLUSTERING_H
