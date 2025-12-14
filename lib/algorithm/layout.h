#ifndef LAYOUT_H
#define LAYOUT_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "../data_structures/graph.h"
#include "../data_structures/clustering.h"

#ifdef NEKOS_HAS_OPENMP
#include <omp.h>
#endif

// 2D point structure
struct Point2D {
    float x;
    float y;

    Point2D() : x(0), y(0) {}
    Point2D(float x_, float y_) : x(x_), y(y_) {}

    Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }

    Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }

    Point2D operator*(float scalar) const {
        return Point2D(x * scalar, y * scalar);
    }

    Point2D operator/(float scalar) const {
        return Point2D(x / scalar, y / scalar);
    }

    float norm() const {
        return std::sqrt(x * x + y * y);
    }

    Point2D normalized() const {
        float n = norm();
        if (n < 1e-6f) return Point2D(0, 0);
        return *this / n;
    }
};

// Force-directed layout using Fruchterman-Reingold algorithm
// Parallelized with OpenMP for better performance
std::vector<Point2D> force_directed_layout(
    const Graph& graph,
    int iterations = 50,
    float k = -1.0f,  // Optimal distance (auto if < 0)
    int num_threads = 1,
    unsigned int seed = 42,
    bool verbose = false
) {
    uint32_t n = graph.num_nodes;

    if (n == 0) {
        return std::vector<Point2D>();
    }

    // Initialize random number generator
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Initialize positions randomly
    std::vector<Point2D> pos(n);
    for (uint32_t i = 0; i < n; i++) {
        pos[i] = Point2D(dist(rng), dist(rng));
    }

    // Optimal distance between nodes
    if (k < 0) {
        k = std::sqrt(1.0f / n);
    }

    float k_squared = k * k;

    // Temperature schedule
    float t = 0.1f;
    float dt = t / (iterations + 1);

    #ifdef NEKOS_HAS_OPENMP
    omp_set_num_threads(num_threads);
    #else
    (void)num_threads;  // Suppress unused parameter warning
    #endif

    if (verbose) {
        std::cout << "Running force-directed layout with " << iterations
                  << " iterations on " << n << " nodes..." << std::endl;
    }

    // Displacement vectors
    std::vector<Point2D> displacement(n);

    // Use grid-based approximation for large graphs (n > 1000)
    bool use_grid = (n > 1000);
    int grid_size = use_grid ? static_cast<int>(std::sqrt(n / 10)) : 0;
    if (grid_size < 10) grid_size = 10;
    if (grid_size > 100) grid_size = 100;

    for (int iter = 0; iter < iterations; iter++) {
        // Reset displacements
        std::fill(displacement.begin(), displacement.end(), Point2D(0, 0));

        if (use_grid) {
            // Grid-based approximation for repulsive forces - O(n) instead of O(n^2)
            std::vector<std::vector<uint32_t>> grid(grid_size * grid_size);

            // Find bounds
            float min_x = pos[0].x, max_x = pos[0].x;
            float min_y = pos[0].y, max_y = pos[0].y;
            for (uint32_t i = 1; i < n; i++) {
                min_x = std::min(min_x, pos[i].x);
                max_x = std::max(max_x, pos[i].x);
                min_y = std::min(min_y, pos[i].y);
                max_y = std::max(max_y, pos[i].y);
            }

            float range_x = max_x - min_x + 1e-6f;
            float range_y = max_y - min_y + 1e-6f;

            // Assign nodes to grid cells
            for (uint32_t i = 0; i < n; i++) {
                int gx = static_cast<int>((pos[i].x - min_x) / range_x * grid_size);
                int gy = static_cast<int>((pos[i].y - min_y) / range_y * grid_size);
                gx = std::max(0, std::min(grid_size - 1, gx));
                gy = std::max(0, std::min(grid_size - 1, gy));
                grid[gy * grid_size + gx].push_back(i);
            }

            // Calculate repulsive forces (only with nearby cells)
            #ifdef NEKOS_HAS_OPENMP
            #pragma omp parallel for schedule(dynamic, 64)
            #endif
            for (uint32_t i = 0; i < n; i++) {
                Point2D disp(0, 0);

                // Find which grid cell this node is in
                int gx = static_cast<int>((pos[i].x - min_x) / range_x * grid_size);
                int gy = static_cast<int>((pos[i].y - min_y) / range_y * grid_size);
                gx = std::max(0, std::min(grid_size - 1, gx));
                gy = std::max(0, std::min(grid_size - 1, gy));

                // Check neighboring cells (3x3 neighborhood)
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = gx + dx;
                        int ny = gy + dy;
                        if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;

                        const auto& cell = grid[ny * grid_size + nx];
                        for (uint32_t j : cell) {
                            if (i == j) continue;

                            Point2D delta = pos[i] - pos[j];
                            float distance = delta.norm();

                            if (distance < 0.01f) {
                                distance = 0.01f;
                            }

                            // Repulsive force: f_r(d) = k^2 / d
                            float repulsive_force = k_squared / distance;
                            disp = disp + delta.normalized() * repulsive_force;
                        }
                    }
                }

                // Add slight attraction to center to prevent peripheral clustering
                Point2D to_center = Point2D(0, 0) - pos[i];
                float dist_to_center = to_center.norm();
                if (dist_to_center > 0.5f) {
                    // Very weak pull toward center, only for far-out nodes
                    disp = disp + to_center.normalized() * (dist_to_center * k * 0.1f);
                }

                displacement[i] = disp;
            }
        } else {
            // Calculate repulsive forces (all pairs) for small graphs
            #ifdef NEKOS_HAS_OPENMP
            #pragma omp parallel for schedule(dynamic, 64)
            #endif
            for (uint32_t i = 0; i < n; i++) {
                Point2D disp(0, 0);

                for (uint32_t j = 0; j < n; j++) {
                    if (i == j) continue;

                    Point2D delta = pos[i] - pos[j];
                    float distance = delta.norm();

                    // Avoid division by zero
                    if (distance < 0.01f) {
                        distance = 0.01f;
                    }

                    // Repulsive force: f_r(d) = k^2 / d
                    float repulsive_force = k_squared / distance;
                    disp = disp + delta.normalized() * repulsive_force;
                }

                displacement[i] = disp;
            }
        }

        // Calculate attractive forces (only for edges)
        // This part is trickier to parallelize due to race conditions
        // We'll use atomic operations or critical sections
        #ifdef NEKOS_HAS_OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
        #endif
        for (uint32_t u = 0; u < n; u++) {
            Point2D local_disp(0, 0);

            uint32_t start = graph.row_ptr[u];
            uint32_t end = graph.row_ptr[u + 1];

            for (uint32_t idx = start; idx < end; idx++) {
                uint32_t v = graph.col_idx[idx];

                // Only process each edge once
                if (u >= v) continue;

                Point2D delta = pos[v] - pos[u];
                float distance = delta.norm();

                if (distance < 0.01f) {
                    distance = 0.01f;
                }

                // Attractive force: f_a(d) = d^2 / k
                float attractive_force = distance / k;
                Point2D force = delta.normalized() * attractive_force;

                local_disp = local_disp + force;

                // Update displacement for v (need atomic or critical)
                #ifdef NEKOS_HAS_OPENMP
                #pragma omp atomic
                displacement[v].x -= force.x;
                #pragma omp atomic
                displacement[v].y -= force.y;
                #else
                displacement[v] = displacement[v] - force;
                #endif
            }

            // Update displacement for u
            #ifdef NEKOS_HAS_OPENMP
            #pragma omp atomic
            displacement[u].x += local_disp.x;
            #pragma omp atomic
            displacement[u].y += local_disp.y;
            #else
            displacement[u] = displacement[u] + local_disp;
            #endif
        }

        // Update positions (limit by temperature)
        #ifdef NEKOS_HAS_OPENMP
        #pragma omp parallel for
        #endif
        for (uint32_t i = 0; i < n; i++) {
            float disp_length = displacement[i].norm();

            if (disp_length < 0.01f) {
                disp_length = 0.1f;
            }

            Point2D delta = displacement[i].normalized() * std::min(disp_length, t);
            pos[i] = pos[i] + delta;
        }

        // Cool temperature
        t -= dt;

        if (verbose && (iter + 1) % 10 == 0) {
            std::cout << "Iteration " << (iter + 1) << "/" << iterations << std::endl;
        }
    }

    // Center positions
    Point2D center(0, 0);
    for (uint32_t i = 0; i < n; i++) {
        center = center + pos[i];
    }
    center = center / n;

    for (uint32_t i = 0; i < n; i++) {
        pos[i] = pos[i] - center;
    }

    // Normalize to [-1, 1]
    float max_coord = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        max_coord = std::max(max_coord, std::abs(pos[i].x));
        max_coord = std::max(max_coord, std::abs(pos[i].y));
    }

    if (max_coord > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            pos[i] = pos[i] / max_coord;
        }
    }

    if (verbose) {
        std::cout << "Layout complete!" << std::endl;
    }

    return pos;
}

// Random layout (for comparison/debugging)
std::vector<Point2D> random_layout(
    const Graph& graph,
    unsigned int seed = 42
) {
    uint32_t n = graph.num_nodes;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<Point2D> pos(n);
    for (uint32_t i = 0; i < n; i++) {
        pos[i] = Point2D(dist(rng), dist(rng));
    }

    return pos;
}

// Clustered force-directed layout
// Groups nodes from the same cluster together using a two-level force approach:
// 1. Intra-cluster forces: Strong attraction between nodes in same cluster
// 2. Inter-cluster forces: Repulsion between different clusters
std::vector<Point2D> clustered_force_directed_layout(
    const Graph& graph,
    const Clustering& clustering,
    int iterations = 100,
    float k = -1.0f,
    int num_threads = 1,
    unsigned int seed = 42,
    bool verbose = false,
    float cluster_separation = 2.0f,
    float intra_cluster_strength = 1.0f)
{
    uint32_t n = graph.num_nodes;

    if (k < 0) {
        k = std::sqrt(1.0f / n);
    }

    #ifdef NEKOS_HAS_OPENMP
    omp_set_num_threads(num_threads);
    #else
    (void)num_threads;
    #endif

    if (verbose) {
        std::cout << "Running clustered force-directed layout:" << std::endl;
        std::cout << "  Nodes: " << n << std::endl;
        std::cout << "  Edges: " << graph.num_edges << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  k: " << k << std::endl;
        std::cout << "  Cluster separation: " << cluster_separation << std::endl;
    }

    // Build cluster membership map (internal node ID -> cluster ID)
    // Store as vector for fast lookup instead of hash map
    std::vector<std::string> node_to_cluster(n);
    std::vector<bool> has_cluster(n, false);
    std::vector<bool> is_singleton(n, false);  // Track if node is in singleton cluster
    for (uint32_t i = 0; i < n; i++) {
        std::string cluster_id = clustering.get_node_cluster(i);
        if (!cluster_id.empty()) {
            node_to_cluster[i] = cluster_id;
            has_cluster[i] = true;
        }
    }

    // Build map of cluster ID -> set of nodes
    std::unordered_map<std::string, std::vector<uint32_t>> cluster_nodes;
    for (uint32_t i = 0; i < n; i++) {
        if (has_cluster[i]) {
            cluster_nodes[node_to_cluster[i]].push_back(i);
        }
    }

    // Mark singleton clusters (only 1 node)
    for (uint32_t i = 0; i < n; i++) {
        if (has_cluster[i]) {
            if (cluster_nodes[node_to_cluster[i]].size() == 1) {
                is_singleton[i] = true;
            }
        }
    }

    if (verbose) {
        int singleton_count = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (is_singleton[i]) singleton_count++;
        }
        std::cout << "  Singleton nodes: " << singleton_count << std::endl;
        std::cout << "  Multi-node cluster nodes: " << (n - singleton_count) << std::endl;
    }

    // Initialize positions: place each cluster in a circular arrangement
    std::vector<Point2D> pos(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Get list of clusters
    std::vector<std::string> cluster_list;
    for (const auto& pair : cluster_nodes) {
        cluster_list.push_back(pair.first);
    }

    // Place clusters in a circle, then place nodes within each cluster
    float num_clusters = static_cast<float>(cluster_list.size());
    for (size_t c = 0; c < cluster_list.size(); c++) {
        const std::string& cluster_id = cluster_list[c];
        const auto& nodes = cluster_nodes[cluster_id];

        // Cluster center position (on a circle)
        float angle = 2.0f * M_PI * c / num_clusters;
        float radius = cluster_separation * num_clusters / (2.0f * M_PI);
        Point2D cluster_center(radius * std::cos(angle), radius * std::sin(angle));

        // Place nodes around cluster center with small random offsets
        for (uint32_t node : nodes) {
            pos[node] = cluster_center + Point2D(dist(rng), dist(rng));
        }
    }

    // Handle unclustered nodes (place them randomly in the center)
    for (uint32_t i = 0; i < n; i++) {
        if (!has_cluster[i]) {
            pos[i] = Point2D(dist(rng), dist(rng));
        }
    }

    float t = 0.1f;  // Initial temperature
    const float dt = t / iterations;

    // Use grid-based approximation for large graphs
    bool use_grid = (n > 1000);
    int grid_size = use_grid ? static_cast<int>(std::sqrt(n / 10)) : 0;
    grid_size = std::max(grid_size, 10);

    // Main iteration loop
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<Point2D> disp(n, Point2D(0, 0));

        if (use_grid) {
            // Grid-based approximation for large graphs
            // Build spatial grid
            std::vector<std::vector<std::vector<uint32_t>>> grid(
                grid_size, std::vector<std::vector<uint32_t>>(grid_size));

            float min_x = pos[0].x, max_x = pos[0].x;
            float min_y = pos[0].y, max_y = pos[0].y;
            for (uint32_t i = 0; i < n; i++) {
                min_x = std::min(min_x, pos[i].x);
                max_x = std::max(max_x, pos[i].x);
                min_y = std::min(min_y, pos[i].y);
                max_y = std::max(max_y, pos[i].y);
            }

            float width = max_x - min_x + 1e-6f;
            float height = max_y - min_y + 1e-6f;

            for (uint32_t i = 0; i < n; i++) {
                int gx = static_cast<int>((pos[i].x - min_x) / width * grid_size);
                int gy = static_cast<int>((pos[i].y - min_y) / height * grid_size);
                gx = std::max(0, std::min(grid_size - 1, gx));
                gy = std::max(0, std::min(grid_size - 1, gy));
                grid[gx][gy].push_back(i);
            }

            // Calculate repulsive forces using grid (only check nearby cells)
            #pragma omp parallel for schedule(dynamic, 64)
            for (uint32_t i = 0; i < n; i++) {
                Point2D local_disp(0, 0);

                int gx = static_cast<int>((pos[i].x - min_x) / width * grid_size);
                int gy = static_cast<int>((pos[i].y - min_y) / height * grid_size);
                gx = std::max(0, std::min(grid_size - 1, gx));
                gy = std::max(0, std::min(grid_size - 1, gy));

                // Check 3x3 neighborhood
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = gx + dx;
                        int ny = gy + dy;
                        if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;

                        for (uint32_t j : grid[nx][ny]) {
                            if (i == j) continue;

                            Point2D delta = pos[i] - pos[j];
                            float dist = delta.norm();
                            if (dist < 1e-6f) {
                                delta = Point2D(0.01f, 0.01f);
                                dist = delta.norm();
                            }

                            // Stronger repulsion between different multi-node clusters
                            // Don't apply enhanced repulsion if either node is a singleton
                            float repulsion_strength = 1.0f;
                            if (has_cluster[i] && has_cluster[j]) {
                                if (node_to_cluster[i] != node_to_cluster[j]) {
                                    // Only enhance repulsion if BOTH nodes are in multi-node clusters
                                    if (!is_singleton[i] && !is_singleton[j]) {
                                        repulsion_strength = cluster_separation;
                                    }
                                }
                            }

                            float repulsive_force = (k * k) / dist * repulsion_strength;
                            local_disp = local_disp + delta.normalized() * repulsive_force;
                        }
                    }
                }

                disp[i] = local_disp;
            }
        } else {
            // Exact O(n^2) calculation for small graphs
            #pragma omp parallel for schedule(dynamic, 64)
            for (uint32_t i = 0; i < n; i++) {
                Point2D local_disp(0, 0);

                // Repulsive forces from all other nodes
                for (uint32_t j = 0; j < n; j++) {
                    if (i == j) continue;

                    Point2D delta = pos[i] - pos[j];
                    float dist = delta.norm();
                    if (dist < 1e-6f) {
                        delta = Point2D(0.01f, 0.01f);
                        dist = delta.norm();
                    }

                    // Stronger repulsion between different multi-node clusters
                    // Don't apply enhanced repulsion if either node is a singleton
                    float repulsion_strength = 1.0f;
                    if (has_cluster[i] && has_cluster[j]) {
                        if (node_to_cluster[i] != node_to_cluster[j]) {
                            // Only enhance repulsion if BOTH nodes are in multi-node clusters
                            if (!is_singleton[i] && !is_singleton[j]) {
                                repulsion_strength = cluster_separation;
                            }
                        }
                    }

                    float repulsive_force = (k * k) / dist * repulsion_strength;
                    local_disp = local_disp + delta.normalized() * repulsive_force;
                }

                disp[i] = local_disp;
            }
        }

        // Calculate attractive forces from edges
        #pragma omp parallel for schedule(dynamic, 64)
        for (uint32_t u = 0; u < n; u++) {
            Point2D local_disp(0, 0);

            for (uint32_t idx = graph.row_ptr[u]; idx < graph.row_ptr[u + 1]; idx++) {
                uint32_t v = graph.col_idx[idx];
                if (v >= u) continue;  // Each edge processed once

                Point2D delta = pos[v] - pos[u];
                float dist = delta.norm();
                if (dist < 1e-6f) continue;

                // Stronger attraction within same cluster (fast vector lookup)
                float attraction_strength = 1.0f;
                if (has_cluster[u] && has_cluster[v]) {
                    if (node_to_cluster[u] == node_to_cluster[v]) {
                        attraction_strength = intra_cluster_strength;
                    }
                }

                float attractive_force = (dist * dist) / k * attraction_strength;
                Point2D force = delta.normalized() * attractive_force;

                #pragma omp atomic
                disp[u].x += force.x;
                #pragma omp atomic
                disp[u].y += force.y;
                #pragma omp atomic
                disp[v].x -= force.x;
                #pragma omp atomic
                disp[v].y -= force.y;
            }
        }

        // Apply displacements
        #pragma omp parallel for
        for (uint32_t i = 0; i < n; i++) {
            float disp_norm = disp[i].norm();
            if (disp_norm > 1e-6f) {
                Point2D displacement = disp[i].normalized() * std::min(disp_norm, t);
                pos[i] = pos[i] + displacement;
            }
        }

        // Cool down
        t -= dt;

        if (verbose && (iter + 1) % 10 == 0) {
            std::cout << "  Iteration " << (iter + 1) << "/" << iterations << std::endl;
        }
    }

    if (verbose) {
        std::cout << "Layout complete!" << std::endl;
    }

    return pos;
}

#endif // LAYOUT_H
