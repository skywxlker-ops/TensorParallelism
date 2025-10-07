#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class Mesh {
public:
    Mesh(int num_gpus);
    ~Mesh();

    int size() const { return num_gpus_; }
    ncclComm_t getComm(int rank) const { return comms_[rank]; }

private:
    int num_gpus_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
};
