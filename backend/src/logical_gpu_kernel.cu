#include <cuda_runtime.h>
#include <iostream>

__global__ void simulateCompute(int logical_id, int* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        printf("[Kernel] Logical GPU %d is running on thread block %d\n", logical_id, blockIdx.x);
    }
    if (idx < 128) output[idx] = logical_id;
}
