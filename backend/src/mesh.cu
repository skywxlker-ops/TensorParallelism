#include "mesh.hpp"

Mesh::Mesh() {
    num_gpus_ = device_count();
    if (num_gpus_ <= 0) {
        throw std::runtime_error("[Mesh] No CUDA devices available.");
    }

    std::cout << "[Mesh] Initializing mesh with " << num_gpus_ << " GPUs..." << std::endl;

    comms_.resize(num_gpus_);
    streams_.resize(num_gpus_);

    std::vector<int> devs(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) {
        devs[i] = i;
        set_device(i, true);
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    NCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, devs.data()));
    std::cout << "[Mesh] NCCL communicators initialized successfully." << std::endl;
}

Mesh::~Mesh() {
    for (int i = 0; i < num_gpus_; ++i) {
        set_device(i, true);
        ncclCommDestroy(comms_[i]);
        cudaStreamDestroy(streams_[i]);
    }
    std::cout << "[Mesh] Destroyed NCCL communicator." << std::endl;
}

void Mesh::allReduce(float* data, int num_elements) const {
    int rank;
    CUDA_CHECK(cudaGetDevice(&rank));

    NCCL_CHECK(ncclAllReduce(
        data, data, num_elements, ncclFloat, ncclSum,
        comms_[rank], streams_[rank]
    ));

    CUDA_CHECK(cudaStreamSynchronize(streams_[rank]));
}
