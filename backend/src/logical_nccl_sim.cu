#include "logical_nccl_sim.hpp"

namespace logical_nccl_sim {
    void simulateAllReduce(float* input,
                           float* output,
                           float* temp,
                           size_t num_elements,
                           cudaStream_t* stream,
                           int physical_id) 
    {
        // naive simulation: just copy input -> output
        for (size_t i = 0; i < num_elements; ++i)
            output[i] = input[i];
    }
}
