#ifndef NEURAL_GRAPH_H
#define NEURAL_GRAPH_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include "graph.h"

#ifdef NEKOS_HAS_OPENMP
#include <omp.h>
#endif

/**
 * NeuralGraph - Graph-defined neural network with OpenMP parallelization
 *
 * Uses graph structure to define network architecture:
 * - Nodes = neurons
 * - Edges = weighted connections
 * - Layer metadata = topological ordering
 *
 * Supports exotic architectures:
 * - Sparse random networks
 * - Small-world networks
 * - Skip connections
 * - Arbitrary DAG topologies
 */
class NeuralGraph {
public:
    // Graph structure (architecture)
    const Graph* graph;

    // Weight matrix (stored as dense for fast computation)
    // weights[i * n_nodes + j] = weight from node j to node i
    std::vector<float> weights;

    // Biases for each node
    std::vector<float> biases;

    // Activations (current state)
    std::vector<float> activations;

    // Gradients (for backprop)
    std::vector<float> gradients;

    // Layer structure (topological order)
    std::vector<std::vector<uint32_t>> layer_order;
    std::vector<uint32_t> node_to_layer;

    // Activation function
    enum ActivationFn {
        TANH,
        RELU,
        SIGMOID,
        LINEAR
    };
    ActivationFn activation_fn;

    // Hyperparameters
    float learning_rate;
    int num_threads;

    // Graph info
    size_t n_nodes;

    /**
     * Constructor
     */
    NeuralGraph(const Graph* g,
                const std::vector<std::vector<uint32_t>>& layers,
                const std::vector<uint32_t>& node_layer_map,
                ActivationFn act_fn = TANH,
                float lr = 0.01f,
                int threads = 1)
        : graph(g),
          layer_order(layers),
          node_to_layer(node_layer_map),
          activation_fn(act_fn),
          learning_rate(lr),
          num_threads(threads),
          n_nodes(g->num_nodes)
    {
        // Initialize weights and biases
        weights.resize(n_nodes * n_nodes, 0.0f);
        biases.resize(n_nodes, 0.0f);
        activations.resize(n_nodes, 0.0f);
        gradients.resize(n_nodes, 0.0f);

        // Xavier initialization for weights
        xavier_init();
    }

    /**
     * Xavier weight initialization
     */
    void xavier_init() {
        float scale = std::sqrt(2.0f / n_nodes);

        #ifdef NEKOS_HAS_OPENMP
        #pragma omp parallel for num_threads(num_threads)
        #endif
        for (size_t i = 0; i < n_nodes; ++i) {
            // Initialize weights for edges that exist in graph
            uint32_t start = graph->row_ptr[i];
            uint32_t end = graph->row_ptr[i + 1];

            for (uint32_t j = start; j < end; ++j) {
                uint32_t neighbor = graph->col_idx[j];
                // Random normal * scale
                float w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
                weights[i * n_nodes + neighbor] = w;
            }

            // Bias initialization
            biases[i] = 0.0f;
        }
    }

    /**
     * Set weight matrix from external source (e.g., PyTorch)
     */
    void set_weights(const std::vector<float>& W) {
        if (W.size() != n_nodes * n_nodes) {
            throw std::invalid_argument("Weight matrix size mismatch");
        }
        weights = W;
    }

    /**
     * Set biases from external source
     */
    void set_biases(const std::vector<float>& b) {
        if (b.size() != n_nodes) {
            throw std::invalid_argument("Bias vector size mismatch");
        }
        biases = b;
    }

    /**
     * Apply activation function
     */
    inline float activate(float x) const {
        switch (activation_fn) {
            case TANH:
                return std::tanh(x);
            case RELU:
                return x > 0.0f ? x : 0.0f;
            case SIGMOID:
                return 1.0f / (1.0f + std::exp(-x));
            case LINEAR:
                return x;
            default:
                return std::tanh(x);
        }
    }

    /**
     * Derivative of activation function
     */
    inline float activate_derivative(float x) const {
        switch (activation_fn) {
            case TANH: {
                float t = std::tanh(x);
                return 1.0f - t * t;
            }
            case RELU:
                return x > 0.0f ? 1.0f : 0.0f;
            case SIGMOID: {
                float s = 1.0f / (1.0f + std::exp(-x));
                return s * (1.0f - s);
            }
            case LINEAR:
                return 1.0f;
            default:
                return 1.0f;
        }
    }

    /**
     * Forward propagation through the network
     * Parallelized across nodes within each layer
     */
    void forward(const std::vector<float>& inputs) {
        if (layer_order.empty()) {
            throw std::runtime_error("Layer order not set");
        }

        if (inputs.size() != layer_order[0].size()) {
            throw std::invalid_argument("Input size mismatch");
        }

        // Clear activations
        std::fill(activations.begin(), activations.end(), 0.0f);

        // Set input layer activations
        const auto& input_nodes = layer_order[0];
        for (size_t i = 0; i < input_nodes.size(); ++i) {
            activations[input_nodes[i]] = inputs[i];
        }

        // Propagate through layers
        for (size_t layer_idx = 1; layer_idx < layer_order.size(); ++layer_idx) {
            const auto& layer_nodes = layer_order[layer_idx];

            // Parallelize across nodes in this layer
            #ifdef NEKOS_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads)
            #endif
            for (size_t i = 0; i < layer_nodes.size(); ++i) {
                uint32_t node = layer_nodes[i];

                // Compute weighted sum: z = sum(w_ij * a_j) + b_i
                float z = biases[node];

                // Sum over all nodes (weights are sparse - only non-zero for edges)
                for (size_t j = 0; j < n_nodes; ++j) {
                    float w = weights[node * n_nodes + j];
                    if (w != 0.0f) {
                        z += w * activations[j];
                    }
                }

                // Apply activation function
                activations[node] = activate(z);
            }
        }
    }

    /**
     * Get output layer activations
     */
    std::vector<float> get_outputs() const {
        if (layer_order.empty()) {
            throw std::runtime_error("Layer order not set");
        }

        const auto& output_nodes = layer_order.back();
        std::vector<float> outputs(output_nodes.size());

        for (size_t i = 0; i < output_nodes.size(); ++i) {
            outputs[i] = activations[output_nodes[i]];
        }

        return outputs;
    }

    /**
     * Backward propagation (MSE loss)
     * Parallelized across nodes within each layer
     */
    float backward(const std::vector<float>& targets) {
        if (layer_order.empty()) {
            throw std::runtime_error("Layer order not set");
        }

        const auto& output_nodes = layer_order.back();
        if (targets.size() != output_nodes.size()) {
            throw std::invalid_argument("Target size mismatch");
        }

        // Clear gradients
        std::fill(gradients.begin(), gradients.end(), 0.0f);

        // Compute output layer gradients (MSE loss)
        float loss = 0.0f;
        for (size_t i = 0; i < output_nodes.size(); ++i) {
            uint32_t node = output_nodes[i];
            float error = activations[node] - targets[i];
            loss += error * error;

            // Gradient = dL/da * da/dz = 2 * error * activation_derivative
            // For simplicity, we store dL/da and multiply by activation_derivative during weight update
            gradients[node] = 2.0f * error;
        }
        loss /= output_nodes.size();

        // Backpropagate through layers (reverse order)
        for (int layer_idx = layer_order.size() - 2; layer_idx >= 0; --layer_idx) {
            const auto& layer_nodes = layer_order[layer_idx];

            // Parallelize across nodes in this layer
            #ifdef NEKOS_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads)
            #endif
            for (size_t i = 0; i < layer_nodes.size(); ++i) {
                uint32_t node = layer_nodes[i];

                // Sum gradients from downstream nodes
                float grad_sum = 0.0f;
                for (size_t j = 0; j < n_nodes; ++j) {
                    // Weight from node -> j
                    float w = weights[j * n_nodes + node];
                    if (w != 0.0f) {
                        grad_sum += w * gradients[j];
                    }
                }

                gradients[node] = grad_sum;
            }
        }

        return loss;
    }

    /**
     * Update weights using gradients (SGD)
     */
    void update_weights() {
        #ifdef NEKOS_HAS_OPENMP
        #pragma omp parallel for num_threads(num_threads)
        #endif
        for (size_t i = 0; i < n_nodes; ++i) {
            // Update bias
            biases[i] -= learning_rate * gradients[i];

            // Update weights
            for (size_t j = 0; j < n_nodes; ++j) {
                if (weights[i * n_nodes + j] != 0.0f) {
                    // Weight gradient = gradient[i] * activation[j]
                    weights[i * n_nodes + j] -= learning_rate * gradients[i] * activations[j];
                }
            }
        }
    }
};

#endif // NEURAL_GRAPH_H
