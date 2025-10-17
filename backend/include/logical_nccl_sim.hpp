#pragma once
#include <cuda_runtime.h>

namespace logical_nccl_sim {
    void simulateAllReduce(float* input,
                           float* output,
                           float* temp,
                           size_t num_elements,
                           cudaStream_t* stream,
                           int physical_id);
}
