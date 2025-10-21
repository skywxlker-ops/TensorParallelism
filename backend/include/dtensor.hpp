#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include "mesh.hpp"

class DTensor {
public:
    DTensor(Mesh* mesh, size_t totalSize);

    void setLayout(const std::vector<std::string>& layout);
    void placeData(const float* host_data);
    void printSlices() const;

private:
    Mesh* mesh_;
    size_t totalSize_;
    std::vector<std::string> layout_;
    std::vector<std::vector<int>> slicesPerGPU_; // [GPU][dim_start, dim_end]
};
