#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include "../data_structures/hierarchical_clustering.h"
// #include "../lib/data_structures/graph.h"
// #include "../lib/io/graph_io.h"
// #include "../lib/algorithm/paris.h"
// #include "../lib/algorithm/champaign.h"

// void print_usage(const char* program_name) {
//     std::cout << "Usage: " << program_name << " <input_file> [options]" << std::endl;
//     std::cout << "\nDescription:" << std::endl;
//     std::cout << "  Run hierarchical clustering algorithm on a graph" << std::endl;
//     std::cout << "\nOptions:" << std::endl;
//     std::cout << "  -a <algorithm>    - Algorithm: champaign or paris (default: champaign)" << std::endl;
//     std::cout << "  -o <output_file>  - Output file for results (default: print to stdout)" << std::endl;
//     std::cout << "  -f <format>       - Output format: json or csv (default: json)" << std::endl;
//     std::cout << "  -t <num_threads>  - Number of threads (default: hardware concurrency)" << std::endl;
//     std::cout << "  -v                - Verbose output" << std::endl;
//     std::cout << "\nAlgorithms:" << std::endl;
//     std::cout << "  champaign  - Size-based distance metric: d = (n_a * n_b) / (total_weight * p(a,b))" << std::endl;
//     // std::cout << "  paris      - Degree-based distance metric: d = p(i) * p(j) / p(i,j) / total_weight" << std::endl;
// }

// }

std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 32) {
                    o << "\\u" << std::hex << std::setfill('0') << std::setw(4) << (int)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

struct TreeNode {
    uint32_t id;
    std::string name;
    std::string type;
    double distance;
    uint32_t count;
    std::vector<TreeNode> children;
};

TreeNode build_tree_from_dendrogram(const std::vector<DendrogramNode>& dendrogram,
                                     uint32_t node_id,
                                     uint32_t num_original_nodes) {
    TreeNode node;
    node.id = node_id;

    if (node_id < num_original_nodes) {
        // Leaf node
        node.name = std::to_string(node_id);
        node.type = "leaf";
        node.count = 1;
        node.distance = 0.0;
    } else {
        // Internal node - find the merge that created this cluster
        uint32_t merge_index = node_id - num_original_nodes;
        if (merge_index < dendrogram.size()) {
            const auto& merge = dendrogram[merge_index];
            node.type = "cluster";
            node.distance = merge.distance;
            node.count = merge.size;

            // Recursively build children
            node.children.push_back(build_tree_from_dendrogram(dendrogram, merge.cluster_a, num_original_nodes));
            node.children.push_back(build_tree_from_dendrogram(dendrogram, merge.cluster_b, num_original_nodes));
        }
    }

    return node;
}

void write_tree_as_json(std::ostream& out, const TreeNode& node, int indent = 0) {
    std::string indent_str(indent * 2, ' ');
    std::string next_indent_str((indent + 1) * 2, ' ');

    out << indent_str << "{\n";
    out << next_indent_str << "\"id\": " << node.id << ",\n";
    out << next_indent_str << "\"type\": \"" << node.type << "\",\n";

    if (node.type == "leaf") {
        out << next_indent_str << "\"name\": \"" << escape_json_string(node.name) << "\",\n";
        out << next_indent_str << "\"count\": " << node.count << "\n";
    } else {
        out << next_indent_str << "\"distance\": ";
        if (std::isinf(node.distance)) {
            out << "Infinity";
        } else {
            out << node.distance;
        }
        out << ",\n";
        out << next_indent_str << "\"count\": " << node.count << ",\n";
        out << next_indent_str << "\"children\": [\n";

        for (size_t i = 0; i < node.children.size(); ++i) {
            write_tree_as_json(out, node.children[i], indent + 2);
            if (i < node.children.size() - 1) {
                out << ",\n";
            } else {
                out << "\n";
            }
        }

        out << next_indent_str << "]\n";
    }

    out << indent_str << "}";
}

void save_dendrogram_json(const std::vector<DendrogramNode>& dendrogram,
                          uint32_t num_nodes,
                          uint32_t num_edges,
                          const std::string& output_file,
                          const std::string& algorithm = "Champaign") {
    // Build the tree from the dendrogram
    // The root is the last merge
    uint32_t root_id = num_nodes + dendrogram.size() - 1;
    TreeNode root = build_tree_from_dendrogram(dendrogram, root_id, num_nodes);

    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }

    out << "{\n";
    out << "  \"algorithm\": \"" << algorithm << "\",\n";
    out << "  \"num_nodes\": " << num_nodes << ",\n";
    out << "  \"num_edges\": " << num_edges << ",\n";
    out << "  \"hierarchy\": ";

    write_tree_as_json(out, root, 1);

    out << "\n}\n";
    out.close();
}

// Helper function to skip whitespace in JSON parsing
inline void skip_whitespace(std::istream& in) {
    while (in && std::isspace(in.peek())) {
        in.get();
    }
}

// Helper function to parse a JSON number (int or double)
inline double parse_json_number(std::istream& in) {
    skip_whitespace(in);
    std::string num_str;
    char c = in.peek();

    // Check for Infinity
    if (c == 'I') {
        std::string word;
        for (int i = 0; i < 8; ++i) {
            if (in) word += in.get();
        }
        if (word == "Infinity") {
            return std::numeric_limits<double>::infinity();
        }
        throw std::runtime_error("Invalid JSON number");
    }

    while (in && (std::isdigit(in.peek()) || in.peek() == '.' || in.peek() == '-' ||
                  in.peek() == 'e' || in.peek() == 'E' || in.peek() == '+')) {
        num_str += in.get();
    }
    return std::stod(num_str);
}

// Helper function to parse a JSON string
inline std::string parse_json_string(std::istream& in) {
    skip_whitespace(in);
    if (in.get() != '"') {
        throw std::runtime_error("Expected '\"' at start of string");
    }

    std::string result;
    while (in) {
        char c = in.get();
        if (c == '"') {
            return result;
        }
        if (c == '\\') {
            c = in.get();
            switch (c) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case 'u': {
                    // Unicode escape - simplified handling
                    char hex[5] = {0};
                    in.read(hex, 4);
                    int code = std::stoi(hex, nullptr, 16);
                    result += static_cast<char>(code);
                    break;
                }
                default: result += c;
            }
        } else {
            result += c;
        }
    }
    throw std::runtime_error("Unterminated string");
}

// Helper function to expect a specific character
inline void expect_char(std::istream& in, char expected) {
    skip_whitespace(in);
    char c = in.get();
    if (c != expected) {
        throw std::runtime_error(std::string("Expected '") + expected + "' but got '" + c + "'");
    }
}

// Forward declaration for recursive parsing
TreeNode parse_tree_node(std::istream& in);

// Parse a JSON object representing a tree node
TreeNode parse_tree_node(std::istream& in) {
    TreeNode node;

    expect_char(in, '{');

    bool first = true;
    while (true) {
        skip_whitespace(in);
        if (in.peek() == '}') {
            in.get();
            break;
        }

        if (!first) {
            expect_char(in, ',');
        }
        first = false;

        std::string key = parse_json_string(in);
        expect_char(in, ':');

        if (key == "id") {
            node.id = static_cast<uint32_t>(parse_json_number(in));
        } else if (key == "name") {
            node.name = parse_json_string(in);
        } else if (key == "type") {
            node.type = parse_json_string(in);
        } else if (key == "distance") {
            node.distance = parse_json_number(in);
        } else if (key == "count") {
            node.count = static_cast<uint32_t>(parse_json_number(in));
        } else if (key == "children") {
            expect_char(in, '[');
            bool first_child = true;
            while (true) {
                skip_whitespace(in);
                if (in.peek() == ']') {
                    in.get();
                    break;
                }
                if (!first_child) {
                    expect_char(in, ',');
                }
                first_child = false;
                node.children.push_back(parse_tree_node(in));
            }
        } else {
            // Skip unknown field
            skip_whitespace(in);
            char c = in.peek();
            if (c == '"') {
                parse_json_string(in);
            } else if (c == '{') {
                parse_tree_node(in);
            } else if (c == '[') {
                in.get();
                int depth = 1;
                while (depth > 0 && in) {
                    c = in.get();
                    if (c == '[') depth++;
                    else if (c == ']') depth--;
                }
            } else {
                parse_json_number(in);
            }
        }
    }

    return node;
}

// Convert TreeNode back to dendrogram format
void tree_to_dendrogram_recursive(const TreeNode& node,
                                   std::vector<DendrogramNode>& dendrogram,
                                   uint32_t num_original_nodes) {
    if (node.type == "cluster" && !node.children.empty()) {
        // Process children first (post-order traversal)
        for (const auto& child : node.children) {
            tree_to_dendrogram_recursive(child, dendrogram, num_original_nodes);
        }

        // Add this merge to the dendrogram
        DendrogramNode dnode;
        dnode.cluster_a = node.children[0].id;
        dnode.cluster_b = node.children[1].id;
        dnode.distance = node.distance;
        dnode.size = node.count;
        dendrogram.push_back(dnode);
    }
}

// Read hierarchical clustering from JSON file
inline std::vector<DendrogramNode> read_dendrogram_json(const std::string& input_file,
                                                         uint32_t& num_nodes,
                                                         uint32_t& num_edges,
                                                         std::string& algorithm) {
    std::ifstream in(input_file);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open input file: " + input_file);
    }

    // Parse the outer JSON object
    expect_char(in, '{');

    TreeNode root;
    bool first = true;

    while (true) {
        skip_whitespace(in);
        if (in.peek() == '}') {
            in.get();
            break;
        }

        if (!first) {
            expect_char(in, ',');
        }
        first = false;

        std::string key = parse_json_string(in);
        expect_char(in, ':');

        if (key == "algorithm") {
            algorithm = parse_json_string(in);
        } else if (key == "num_nodes") {
            num_nodes = static_cast<uint32_t>(parse_json_number(in));
        } else if (key == "num_edges") {
            num_edges = static_cast<uint32_t>(parse_json_number(in));
        } else if (key == "hierarchy") {
            root = parse_tree_node(in);
        }
    }

    // Convert tree back to dendrogram
    std::vector<DendrogramNode> dendrogram;
    tree_to_dendrogram_recursive(root, dendrogram, num_nodes);

    return dendrogram;
}
