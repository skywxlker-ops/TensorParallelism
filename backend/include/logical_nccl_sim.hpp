#pragma once
#include <cuda_runtime.h>

namespace logical_nccl_sim {

// Declaration only
void simulateAllReduce(float* A, float* B, float* C, size_t N, cudaStream_t stream, int logical_id);

} // namespace logical_nccl_sim
