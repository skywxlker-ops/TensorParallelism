#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

namespace logical_nccl_sim {
    void simulateAllReduce(float* sendbuf, float* recvbuf, float* tmpbuf,
                           unsigned long count, cudaStream_t* stream, int rank);
}

class Mesh {
public:
    Mesh(int num_physical, int logical_per_physical);
    ~Mesh();

    void allReduce(std::vector<float*>& buffers, int num_elements);

private:
    int num_physical_;
    int logical_per_physical_;
    int num_logical_;
    std::vector<int> logical_to_physical_;
    std::vector<cudaStream_t> streams_;
};
