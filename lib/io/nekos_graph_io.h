#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <chrono>
#include <algorithm>  // For std::sort

#ifdef NEKOS_HAS_OPENMP
#include <omp.h>
#endif

#include "../data_structures/graph.h"

#include "mapped_file.h"

Graph load_undirected_tsv_edgelist_parallel(const std::string& filename, int num_threads = std::thread::hardware_concurrency(), bool verbose = false, bool skip_header = false, char delimiter = '\t') {
    Graph graph;
    MappedFile file;

    if (!file.open(filename)) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return graph;
    }

    #ifdef NEKOS_HAS_OPENMP
    omp_set_num_threads(num_threads);
    #else
    (void)num_threads;  // Suppress unused parameter warning
    #endif

    const char* data = file.data();
    size_t file_size = file.size();
    size_t data_start = 0;

    // Skip header line if requested
    if (skip_header && file_size > 0) {
        while (data_start < file_size && data[data_start] != '\n') {
            data_start++;
        }
        if (data_start < file_size) {
            data_start++;  // Skip the newline
        }
        if (verbose) {
            std::cout << "Skipping header line (first " << data_start << " bytes)" << std::endl;
        }
    }

    if (verbose) {
        std::cout << "File size: " << file_size / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Step 1: Parsing file and collecting edges..." << std::endl;
    }

    // Step 1: Parse file in parallel chunks (starting after header if skipped)
    size_t effective_size = file_size - data_start;
    size_t chunk_size = 64 * 1024 * 1024; // 64 MB chunks
    size_t num_chunks = (effective_size + chunk_size - 1) / chunk_size;
    
    #ifdef NEKOS_HAS_OPENMP
    int actual_threads = num_threads;
    #else
    int actual_threads = 1;
    #endif

    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> thread_edges(actual_threads);
    std::vector<std::unordered_set<uint64_t>> thread_nodes(actual_threads);

    #ifdef NEKOS_HAS_OPENMP
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_edges[thread_id].reserve(1000000);

        #pragma omp for schedule(dynamic)
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    #else
    {
        int thread_id = 0;
        thread_edges[thread_id].reserve(1000000);

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    #endif
            size_t chunk_begin = data_start + (chunk_idx * chunk_size);
            size_t chunk_end = std::min(chunk_begin + chunk_size, file_size);

            // Adjust chunk_begin to start at beginning of a line (except first chunk after header)
            if (chunk_begin > data_start) {
                while (chunk_begin < file_size && data[chunk_begin-1] != '\n') {
                    chunk_begin++;
                }
            }
            
            // Adjust chunk_end to end at end of a complete line (except last chunk)
            if (chunk_end < file_size) {
                while (chunk_end < file_size && data[chunk_end] != '\n') {
                    chunk_end++;
                }
                if (chunk_end < file_size) chunk_end++; // Include the newline
            }
            
            // Skip empty chunks that might result from boundary adjustments
            if (chunk_begin >= chunk_end) continue;
            
            // Process the chunk
            const char* ptr = data + chunk_begin;
            const char* end = data + chunk_end;
            
            while (ptr < end) {
                // Parse source node
                uint64_t src = 0;
                while (ptr < end && *ptr >= '0' && *ptr <= '9') {
                    src = src * 10 + (*ptr - '0');
                    ptr++;
                }

                // Skip delimiter
                if (ptr < end && *ptr == delimiter) ptr++;

                // Parse target node
                uint64_t dst = 0;
                while (ptr < end && *ptr >= '0' && *ptr <= '9') {
                    dst = dst * 10 + (*ptr - '0');
                    ptr++;
                }
                
                // Store edge and nodes
                thread_edges[thread_id].emplace_back(src, dst);
                thread_nodes[thread_id].insert(src);
                thread_nodes[thread_id].insert(dst);
                
                // Skip to next line
                while (ptr < end && *ptr != '\n') ptr++;
                if (ptr < end) ptr++; // Skip newline
            }
        }
    }
    
    if (verbose) {
        std::cout << "Step 2: Building global node mapping..." << std::endl;
    }
    
    // Merge thread-local node sets
    std::unordered_set<uint64_t> all_nodes_set;
    for (const auto& nodes : thread_nodes) {
        all_nodes_set.insert(nodes.begin(), nodes.end());
    }
    thread_nodes.clear(); // Free memory
    
    // CRITICAL FIX: Convert to sorted vector for deterministic ordering
    std::vector<uint64_t> all_nodes(all_nodes_set.begin(), all_nodes_set.end());
    std::sort(all_nodes.begin(), all_nodes.end()); // Sort to ensure deterministic mapping
    all_nodes_set.clear(); // Free memory
    
    // Build node mapping with deterministic order
    graph.num_nodes = all_nodes.size();
    graph.node_map.reserve(graph.num_nodes);
    graph.id_map.resize(graph.num_nodes);
    
    if (verbose) {
        std::cout << "Building deterministic node mapping for " << graph.num_nodes << " nodes..." << std::endl;
        if (graph.num_nodes > 0) {
            std::cout << "Node ID range: " << all_nodes.front() << " to " << all_nodes.back() << std::endl;
        }
    }
    
    uint32_t next_id = 0;
    for (uint64_t original_id : all_nodes) { // Now iterating in sorted order!
        graph.node_map[original_id] = next_id;
        graph.id_map[next_id] = original_id;
        next_id++;
    }
    all_nodes.clear(); // Free memory
    
    // Count total edges
    for (const auto& edges : thread_edges) {
        graph.num_edges += edges.size();
    }

    if (verbose) {
        std::cout << "Found " << graph.num_nodes << " nodes and "
                  << graph.num_edges << " undirected edges." << std::endl;
        std::cout << "Step 3: Counting degrees..." << std::endl;
    }

    // Step 3: Count degrees in parallel
    std::vector<uint32_t> degree(graph.num_nodes, 0);

    #ifdef NEKOS_HAS_OPENMP
    #pragma omp parallel for
    #endif
    for (int t = 0; t < actual_threads; ++t) {
        std::vector<uint32_t> local_degree(graph.num_nodes, 0);
        
        for (const auto& edge : thread_edges[t]) {
            uint32_t src_id = graph.node_map[edge.first];
            uint32_t dst_id = graph.node_map[edge.second];
            local_degree[src_id]++;
            local_degree[dst_id]++;
        }

        // Merge local degrees into global degree array
        #ifdef NEKOS_HAS_OPENMP
        #pragma omp critical
        #endif
        {
            for (uint32_t i = 0; i < graph.num_nodes; ++i) {
                degree[i] += local_degree[i];
            }
        }
    }
    
    if (verbose) {
        std::cout << "Step 4: Building CSR structure..." << std::endl;
    }
    
    // Build row pointers
    graph.row_ptr.resize(graph.num_nodes + 1);
    graph.row_ptr[0] = 0;
    for (uint32_t i = 0; i < graph.num_nodes; ++i) {
        graph.row_ptr[i + 1] = graph.row_ptr[i] + degree[i];
    }
    
    // Allocate column indices
    size_t total_directed_edges = graph.row_ptr.back();
    graph.col_idx.resize(total_directed_edges);
    
    // Use degree array as offset tracker (reset to 0)
    std::fill(degree.begin(), degree.end(), 0);
    
    // Fill CSR structure - this needs to be sequential or use atomic operations
    // Option 1: Sequential (simpler and often fast enough)
    for (int t = 0; t < actual_threads; ++t) {
        for (const auto& edge : thread_edges[t]) {
            uint32_t src_id = graph.node_map[edge.first];
            uint32_t dst_id = graph.node_map[edge.second];
            
            // Add edge src -> dst
            graph.col_idx[graph.row_ptr[src_id] + degree[src_id]++] = dst_id;
            
            // Add edge dst -> src (undirected)
            graph.col_idx[graph.row_ptr[dst_id] + degree[dst_id]++] = src_id;
        }
    }
    
    thread_edges.clear(); // Free memory
    
    if (verbose) {
        std::cout << "Loaded undirected graph with " << graph.num_nodes << " nodes and " 
                  << graph.num_edges << " edges (" << total_directed_edges << " directed edges in CSR)" << std::endl;
    }
    
    return graph;
}

// Save graph to a TSV edgelist file
bool save_graph_edgelist(const std::string& filename, const Graph& graph, bool verbose = false) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }
    
    if (verbose) {
        std::cout << "Saving graph to: " << filename << std::endl;
    }
    
    size_t edges_written = 0;
    size_t total_edges = graph.num_edges;
    
    // For each node, write its edges (but only in one direction to avoid duplicates)
    for (uint32_t node_id = 0; node_id < graph.num_nodes; ++node_id) {
        uint64_t original_node_id = graph.id_map[node_id];
        
        for (uint32_t i = graph.row_ptr[node_id]; i < graph.row_ptr[node_id + 1]; ++i) {
            uint32_t neighbor_id = graph.col_idx[i];
            
            // Only write edges where node_id < neighbor_id to avoid duplicates
            if (node_id < neighbor_id) {
                uint64_t original_neighbor_id = graph.id_map[neighbor_id];
                outfile << original_node_id << "\t" << original_neighbor_id << "\n";
                
                edges_written++;
                if (verbose && edges_written % 1000000 == 0) {
                    double progress = 100.0 * edges_written / total_edges;
                    std::cout << "\rSaving edges: " << std::fixed << std::setprecision(1) 
                              << progress << "%" << std::flush;
                }
            }
        }
    }
    
    if (verbose) {
        std::cout << std::endl; // End progress line
        std::cout << "Saved " << edges_written << " edges" << std::endl;
    }
    
    outfile.close();
    return true;
}

#endif // GRAPH_IO_H
