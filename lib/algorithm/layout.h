#ifndef LAYOUT_H
#define LAYOUT_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "../data_structures/graph.h"

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

#endif // LAYOUT_H
