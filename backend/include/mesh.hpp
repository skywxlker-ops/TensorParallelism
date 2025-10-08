#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "cudafunctions.hpp"

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class Mesh {
public:
    Mesh();   // auto-detect GPUs
    ~Mesh();

    int size() const { return num_gpus_; }
    ncclComm_t getComm(int rank) const { return comms_[rank]; }
    cudaStream_t getStream(int rank) const { return streams_[rank]; }

    void allReduce(float* data, int num_elements) const;

private:
    int num_gpus_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
};
