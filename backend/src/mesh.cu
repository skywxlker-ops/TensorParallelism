#include "mesh.hpp"
#include <numeric>

Mesh::Mesh(int num_logical, int buffer_size)
    : total_logical_(num_logical), buffer_size_(buffer_size)
{
    buffers_.resize(total_logical_);
    for(int i = 0; i < total_logical_; i++) {
        buffers_[i].resize(buffer_size_);
        for(int j = 0; j < buffer_size_; j++)
            buffers_[i][j] = float(i + j);  // initialize with sample values
    }
    std::cout << "[Mesh] Initialized " << total_logical_ << " logical GPUs.\n";
}

Mesh::~Mesh() {
    // vectors clean themselves
}

void Mesh::simulateAllReduce() {
    std::vector<float> result(buffer_size_, 0.0f);
    for(int i = 0; i < total_logical_; i++)
        for(int j = 0; j < buffer_size_; j++)
            result[j] += buffers_[i][j];

    // Scatter back
    for(int i = 0; i < total_logical_; i++)
        for(int j = 0; j < buffer_size_; j++)
            buffers_[i][j] = result[j];

    std::cout << "[Mesh] Simulated intra-physical AllReduce done.\n";
}
