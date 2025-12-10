#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <set>
#include <atomic>

#ifdef NEKOS_HAS_OPENMP
#include <omp.h>
#endif

// CSR representation for undirected graph
struct Graph {
    // Graph id
    std::string id;

    std::vector<uint32_t> row_ptr;  // Offsets for each node's edge list
    std::vector<uint32_t> col_idx;  // Target nodes
    
    // Node ID mapping (if needed)
    std::unordered_map<uint64_t, uint32_t> node_map;
    std::vector<uint64_t> id_map;
    
    // Graph info
    size_t num_nodes = 0;
    size_t num_edges = 0; // This counts each undirected edge once

    // Add an edge to the graph
    void add_edge(uint32_t from, uint32_t to) {
        // Check if nodes exist
        if (from >= num_nodes || to >= num_nodes) {
            return;
        }
        
        // Check if edge already exists
        for (uint32_t i = row_ptr[from]; i < row_ptr[from + 1]; ++i) {
            if (col_idx[i] == to) {
                return; // Edge already exists
            }
        }
        
        // We need to shift all row pointers after 'from' to accommodate the new edge
        for (uint32_t i = from + 1; i <= num_nodes; ++i) {
            row_ptr[i]++;
        }
        
        // Insert the new edge
        col_idx.insert(col_idx.begin() + row_ptr[from], to);
        
        // Add the reverse edge (for undirected graph)
        for (uint32_t i = to + 1; i <= num_nodes; ++i) {
            row_ptr[i]++;
        }
        
        col_idx.insert(col_idx.begin() + row_ptr[to], from);
        
        // Update edge count
        num_edges++;
    }

    // Add a node to the graph
    uint32_t add_node(uint64_t original_id) {
        uint32_t new_id = num_nodes;
        node_map[original_id] = new_id;
        id_map.push_back(original_id);
        num_nodes++;
        row_ptr.push_back(row_ptr.back()); // Initialize new row_ptr entry
        return new_id;
    }

    // Get degree of a node
    uint32_t get_degree(uint32_t node) const {
        if (node >= num_nodes) return 0;
        return row_ptr[node + 1] - row_ptr[node];
    }

    // Get neighbors of a node
    std::vector<uint32_t> get_neighbors(uint32_t node) const {
        if (node >= num_nodes) return {};
        uint32_t start = row_ptr[node];
        uint32_t end = row_ptr[node + 1];
        return std::vector<uint32_t>(col_idx.begin() + start, col_idx.begin() + end);
    }
};

// Sort adjacency lists in parallel
void sort_adjacency_lists_parallel(Graph& graph, int num_threads, bool verbose = false) {
    if (verbose) {
        std::cout << "Sorting adjacency lists..." << std::endl;
    }

    #ifdef NEKOS_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t i = 0; i < graph.num_nodes; i++) {
        uint32_t start = graph.row_ptr[i];
        uint32_t end = graph.row_ptr[i + 1];

        if (end > start) {
            std::sort(&graph.col_idx[start], &graph.col_idx[end]);
        }
    }
}

// Remove self-loops and duplicate edges in parallel
void clean_graph_parallel(Graph& graph, int num_threads, bool verbose = false) {
    if (verbose) {
        std::cout << "Removing self-loops and duplicate edges..." << std::endl;
    }

    // First, sort adjacency lists to place duplicates adjacent to each other
    sort_adjacency_lists_parallel(graph, num_threads, verbose);

    // Count unique edges for each vertex
    std::vector<uint32_t> unique_counts(graph.num_nodes, 0);

    #ifdef NEKOS_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t i = 0; i < graph.num_nodes; i++) {
        uint32_t start = graph.row_ptr[i];
        uint32_t end = graph.row_ptr[i + 1];
        
        if (start == end) continue; // No edges
        
        uint32_t last = UINT32_MAX; // Initialize to an impossible node ID
        uint32_t count = 0;
        
        for (uint32_t j = start; j < end; j++) {
            uint32_t neighbor = graph.col_idx[j];
            
            // Skip self-loops and duplicates
            if (neighbor != i && neighbor != last) {
                count++;
                last = neighbor;
            }
        }
        
        unique_counts[i] = count;
    }
    
    // Compute new row pointers based on unique counts
    std::vector<uint32_t> new_row_ptr(graph.num_nodes + 1);
    new_row_ptr[0] = 0;
    
    for (size_t i = 0; i < graph.num_nodes; i++) {
        new_row_ptr[i + 1] = new_row_ptr[i] + unique_counts[i];
    }
    
    // Allocate new column indices
    std::vector<uint32_t> new_col_idx(new_row_ptr.back());

    // Fill new column indices
    #ifdef NEKOS_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
    for (size_t i = 0; i < graph.num_nodes; i++) {
        uint32_t start = graph.row_ptr[i];
        uint32_t end = graph.row_ptr[i + 1];
        uint32_t new_start = new_row_ptr[i];
        
        if (start == end) continue; // No edges
        
        uint32_t last = UINT32_MAX; // Initialize to an impossible node ID
        uint32_t pos = new_start;
        
        for (uint32_t j = start; j < end; j++) {
            uint32_t neighbor = graph.col_idx[j];
            
            // Skip self-loops and duplicates
            if (neighbor != i && neighbor != last) {
                new_col_idx[pos++] = neighbor;
                last = neighbor;
            }
        }
    }
    
    size_t removed_edges = (graph.col_idx.size() - new_col_idx.size()) / 2;
    
    // Update the graph
    graph.col_idx = std::move(new_col_idx);
    graph.row_ptr = std::move(new_row_ptr);
    
    if (verbose) {
        std::cout << "Removed " << removed_edges << " edges (self-loops and duplicates)" << std::endl;
    }
}

void add_edges_batch(Graph& g, const std::vector<std::pair<uint32_t, uint32_t>>& edges_to_add) {
    if (edges_to_add.empty()) return;
    
    // Create temporary adjacency structure using sets to avoid duplicates
    std::vector<std::set<uint32_t>> temp_adj(g.num_nodes);
    
    // First, add all existing edges from CSR to temp structure
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            temp_adj[u].insert(v);
        }
    }
    
    // Add new edges (both directions for undirected graph)
    size_t new_edges_added = 0;
    for (const auto& [u, v] : edges_to_add) {
        if (u >= g.num_nodes || v >= g.num_nodes || u == v) continue;
        
        // Check if edge already exists
        if (temp_adj[u].count(v) == 0) {
            temp_adj[u].insert(v);
            temp_adj[v].insert(u);
            new_edges_added++;
        }
    }
    
    // Count total edges for new CSR
    size_t total_directed_edges = 0;
    for (const auto& neighbors : temp_adj) {
        total_directed_edges += neighbors.size();
    }
    
    // Rebuild CSR structure
    std::vector<uint32_t> new_row_ptr(g.num_nodes + 1);
    std::vector<uint32_t> new_col_idx;
    new_col_idx.reserve(total_directed_edges);
    
    new_row_ptr[0] = 0;
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        // Add all neighbors in sorted order (set maintains order)
        for (uint32_t v : temp_adj[u]) {
            new_col_idx.push_back(v);
        }
        new_row_ptr[u + 1] = new_col_idx.size();
    }
    
    // Update graph structure
    g.row_ptr = std::move(new_row_ptr);
    g.col_idx = std::move(new_col_idx);
    g.num_edges += new_edges_added;
    
    std::cout << "Batch added " << new_edges_added << " new edges" << std::endl;
}

std::vector<std::tuple<uint32_t, uint32_t>> get_edges(const Graph& g) {
    std::vector<std::tuple<uint32_t, uint32_t>> edges;
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            if (u < v) { // Avoid duplicates in undirected graph
                edges.emplace_back(u, v);
            }
        }
    }
    return edges;
}

#endif // GRAPH_H
