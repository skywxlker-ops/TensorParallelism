#include "logical_nccl_sim.hpp"
#include <iostream>
#include <cuda_runtime.h>

// Kernel
__global__ void allReduceKernel(const float* A, const float* B, float* C, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Wrapper
namespace logical_nccl_sim {

void simulateAllReduce(float* A, float* B, float* C, size_t N, cudaStream_t stream, int logical_id) {
    // Ensure correct device is set based on pointer
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, A);
    cudaSetDevice(attr.device);

    // Debug: check first few values of A and B
    float hostA[5], hostB[5];
    cudaMemcpy(hostA, A, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostB, B, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[SimNCCL Debug] Logical GPU " << logical_id << " pre-kernel A[0..4]: ";
    for (int i = 0; i < 5; ++i) std::cout << hostA[i] << " ";
    std::cout << "\n[SimNCCL Debug] Logical GPU " << logical_id << " pre-kernel B[0..4]: ";
    for (int i = 0; i < 5; ++i) std::cout << hostB[i] << " ";
    std::cout << std::endl;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    allReduceKernel<<<blocks, threads, 0, stream>>>(A, B, C, N);
    cudaStreamSynchronize(stream);

    float host_buf[5];
    cudaMemcpy(host_buf, C, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[SimNCCL] Logical GPU " << logical_id 
              << " first 5 elements after simulated AllReduce: ";
    for (int i = 0; i < 5; ++i) std::cout << host_buf[i] << " ";
    std::cout << std::endl;
}

} // namespace logical_nccl_sim

