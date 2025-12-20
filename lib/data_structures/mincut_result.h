#include <vector>
#include <cstdint>

/** 
 * Container for mincut output
 * 
 * @param light_partition the nodes on one side of the mincut
 * @param heavy_partition the nodes on the other side of the mincut
 * @param cut_size the number of edges in the cut
 */
class MincutResult {
    std::vector<uint32_t> light_partition;
    std::vector<uint32_t> heavy_partition;
    uint32_t cut_size;

public:
    MincutResult(std::vector<uint32_t> light_, 
                 std::vector<uint32_t> heavy_,
                 uint32_t cut_) : light_partition(light_), heavy_partition(heavy_), cut_size(cut_) {}

    MincutResult(std::vector<int> light_, 
                 std::vector<int> heavy_,
                 int cut_) :
                 light_partition(light_.begin(), light_.end()),
                 heavy_partition(heavy_.begin(), heavy_.end()),
                 cut_size(static_cast<uint32_t>(cut_)) {}

    std::vector<uint32_t> get_light_partition() const { return light_partition; }
    std::vector<uint32_t> get_heavy_partition() const { return heavy_partition; }
    uint32_t get_cut_size() const { return cut_size; }
};
