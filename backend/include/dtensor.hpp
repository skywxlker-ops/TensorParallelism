#pragma once
#include "mesh.hpp"
#include <vector>
#include <string>
#include <map>

class DTensor {
public:
    DTensor(const std::vector<int64_t>& shape, Mesh& mesh);

    void setLayout(const std::vector<std::string>& layout);
    void placeData(const float* host_data);
    void printHostTensor() const;
    void printSlices() const;

private:
    std::vector<int64_t> shape_;                     // Global tensor shape
    std::vector<std::string> layout_;                // Per-dimension placement: shard / replicate / partial
    Mesh& mesh_;
    std::map<int, std::vector<std::pair<int,int>>> slices_; // [start, end] per dim per GPU

    // helper
    std::vector<int64_t> computeLocalStartEnd(int dim, int coord, int mesh_dim_size) const;
};
