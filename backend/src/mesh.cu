#include "mesh.hpp"
#include <iostream>

Mesh::Mesh(int numPhysical, int logicalsPerPhysical, int bufSize)
    : num_physical_(numPhysical), logicals_per_phys_(logicalsPerPhysical),
      buffer_size_(bufSize) 
{
    total_logical_ = num_physical_ * logicals_per_phys_;
    buffers_.resize(total_logical_);

    for (int i = 0; i < total_logical_; ++i) {
        cudaMalloc(&buffers_[i], buffer_size_ * sizeof(float));
        cudaMemset(buffers_[i], 0, buffer_size_ * sizeof(float));
    }

    std::cout << "[Mesh] Initialized " << total_logical_ << " logical GPUs." << std::endl;
}

Mesh::~Mesh() {
    for (auto buf : buffers_) {
        cudaFree(buf);
    }
}
