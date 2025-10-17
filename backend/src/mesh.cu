#include "mesh.hpp"
#include "logical_nccl_sim.hpp"

Mesh::Mesh(int num_physical, int logical_per_physical)
    : num_physical_(num_physical),
      logical_per_physical_(logical_per_physical),
      num_logical_(num_physical * logical_per_physical)
{
    std::cout << "[Mesh] Initializing mesh with " << num_physical_ << " GPUs...\n";

    // map logical -> physical
    for (int i = 0; i < num_logical_; i++)
        logical_to_physical_.push_back(i % num_physical_);

    // create streams
    streams_.resize(num_logical_);
    for (int i = 0; i < num_logical_; i++)
        cudaStreamCreate(&streams_[i]);
    
    std::cout << "[Mesh] Initialized " << num_logical_ << " logical GPUs across "
              << num_physical_ << " physical GPUs.\n";
}

Mesh::~Mesh() {
    for (auto& s : streams_)
        cudaStreamDestroy(s);
}

void Mesh::allReduce(std::vector<float*>& buffers, int num_elements) {
    for (int i = 0; i < num_logical_; i++) {
        logical_nccl_sim::simulateAllReduce(
            buffers[i],   // send
            buffers[i],   // recv
            buffers[i],   // tmp
            num_elements,
            &streams_[i],
            logical_to_physical_[i]
        );
        std::cout << "[SimNCCL] Logical GPU " << i << " completed AllReduce.\n";
    }
}
