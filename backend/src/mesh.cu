#include "mesh.hpp"

Mesh::Mesh(int num_gpus) : num_gpus_(num_gpus) {
    std::cout << "[Mesh] Initializing mesh with " << num_gpus_ << " GPUs..." << std::endl;
    comms_.resize(num_gpus_);
    streams_.resize(num_gpus_);

    std::vector<int> devs(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) {
        devs[i] = i;
        cudaSetDevice(i);
        cudaStreamCreate(&streams_[i]);
    }

    NCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, devs.data()));
    std::cout << "[Mesh] NCCL communicators initialized successfully." << std::endl;
}

Mesh::~Mesh() {
    for (int i = 0; i < num_gpus_; ++i) {
        cudaSetDevice(i);
        ncclCommDestroy(comms_[i]);
        cudaStreamDestroy(streams_[i]);
    }
    std::cout << "[Mesh] Destroyed NCCL communicator." << std::endl;
}
