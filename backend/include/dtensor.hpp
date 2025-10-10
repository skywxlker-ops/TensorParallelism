#pragma once
#include "mesh.hpp"
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <iostream>

class DTensor {
public:
    DTensor(const std::vector<int64_t>& shape, Mesh& mesh);
    ~DTensor();

    void setLayout(const std::vector<std::string>& layout);
    std::pair<int64_t,int64_t> getSliceForDim(int dim, int gpu_id) const;
    void printSlices() const;

    float* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }

private:
    float* data_ = nullptr;
    std::vector<int64_t> shape_;
    size_t size_;
    Mesh& mesh_;
    std::vector<std::string> layout_;
};
