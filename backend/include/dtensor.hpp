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

    // GPU initialization
    void initOnGPU(float* d_data, float value, int64_t n);

private:
    std::vector<int64_t> shape_;
    std::vector<std::string> layout_;
    Mesh& mesh_;

    // slices_[gpu][dim] = {start, end}
    std::map<int, std::vector<std::pair<int64_t,int64_t>>> slices_;

    void computeSlices();
};
