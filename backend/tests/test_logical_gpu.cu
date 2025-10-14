#include "logical_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>

// Declare kernel
__global__ void simulateCompute(int logical_id, int* output);

int main() {
    LogicalGPUManager manager;
    size_t mem_per_logical = 256 * 1024 * 1024;
    manager.init(2, mem_per_logical); // 2 logical per physical
    manager.printInfo();

    const auto& logicals = manager.getLogicalGPUs();

    // Launch a kernel per logical GPU
    for (const auto& lgpu : logicals) {
        cudaSetDevice(lgpu.physical_id);
        int* d_output = nullptr;
        cudaMallocAsync(&d_output, 128 * sizeof(int), lgpu.stream);

        std::cout << "[Launch] Kernel on Logical GPU " << lgpu.logical_id
                  << " (Physical " << lgpu.physical_id << ")" << std::endl;

        simulateCompute<<<1, 128, 0, lgpu.stream>>>(lgpu.logical_id, d_output);
    }

    // Sync all streams
    for (const auto& lgpu : logicals) {
        cudaSetDevice(lgpu.physical_id);
        cudaStreamSynchronize(lgpu.stream);
    }

    std::cout << "\nAll logical GPU kernels finished!\n" << std::endl;
    return 0;
}
