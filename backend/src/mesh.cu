// #include "mesh.hpp"

// Mesh::Mesh(int num_gpus) : num_gpus_(num_gpus) {
//     std::cout << "[Mesh] Initializing mesh with " << num_gpus_ << " GPUs..." << std::endl;
//     comms_.resize(num_gpus_);
//     streams_.resize(num_gpus_);

//     std::vector<int> devs(num_gpus_);
//     for (int i = 0; i < num_gpus_; ++i) {
//         devs[i] = i;
//         cudaSetDevice(i);
//         cudaStreamCreate(&streams_[i]);
//     }

//     NCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, devs.data()));
//     std::cout << "[Mesh] NCCL communicators initialized successfully." << std::endl;
// }

// Mesh::~Mesh() {
//     for (int i = 0; i < num_gpus_; ++i) {
//         cudaSetDevice(i);
//         ncclCommDestroy(comms_[i]);
//         cudaStreamDestroy(streams_[i]);
//     }
//     std::cout << "[Mesh] Destroyed NCCL communicator." << std::endl;
// }


#include "mesh.hpp"

Mesh::Mesh(int num_gpus) : num_gpus_(num_gpus) {
    std::cout << "[Mesh] Initializing mesh with " << num_gpus_ << " GPUs..." << std::endl;

    comms_.resize(num_gpus_);
    streams_.resize(num_gpus_);

    std::vector<int> devs(num_gpus_);
    for (int i = 0; i < num_gpus_; ++i) {
        devs[i] = i;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    NCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, devs.data()));
    std::cout << "[Mesh] NCCL communicators initialized successfully." << std::endl;
}

Mesh::~Mesh() {
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        ncclCommDestroy(comms_[i]);
        cudaStreamDestroy(streams_[i]);
    }
    std::cout << "[Mesh] Destroyed NCCL communicator." << std::endl;
}

void Mesh::allReduce(float* data, int num_elements) const {
    int rank;
    CUDA_CHECK(cudaGetDevice(&rank)); // current GPU

    NCCL_CHECK(ncclAllReduce(
        data, data, num_elements, ncclFloat, ncclSum,
        comms_[rank], 0 // default stream
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
}


