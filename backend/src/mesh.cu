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

        // Fill each buffer with a unique sequence (i, i+1, ...)
        std::vector<float> temp(bufSize);
        for (int j = 0; j < bufSize; ++j) temp[j] = i + j;
        cudaMemcpy(buffers_[i], temp.data(), bufSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "[Mesh] Initialized " << total_logical_ << " logical GPUs." << std::endl;
}

Mesh::~Mesh() {
    for (auto buf : buffers_) {
        cudaFree(buf);
    }
}
