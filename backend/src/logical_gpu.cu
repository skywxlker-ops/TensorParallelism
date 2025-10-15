#include "logical_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>

void LogicalGPUManager::init(int logical_per_phys, size_t N_per_buffer) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[LogicalGPUManager] No physical GPUs found!\n";
        return;
    }

    N_per_buffer_ = N_per_buffer;
    std::cout << "[LogicalGPU] Initializing " << device_count * logical_per_phys 
              << " logical GPUs across " << device_count << " physical GPUs.\n";

    for (int d = 0; d < device_count; ++d) {
        cudaSetDevice(d);
        for (int l = 0; l < logical_per_phys; ++l) {
            LogicalGPU lgpu;
            lgpu.physical_id = d;
            lgpu.logical_id = d * logical_per_phys + l;
            cudaStreamCreate(&lgpu.stream);

            // Allocate separate buffers per logical GPU
            cudaSetDevice(d);
            cudaMalloc(&lgpu.bufA, N_per_buffer_ * sizeof(float));
            cudaMalloc(&lgpu.bufB, N_per_buffer_ * sizeof(float));
            cudaMalloc(&lgpu.bufC, N_per_buffer_ * sizeof(float));

            // Initialize buffers
            std::vector<float> host_init(N_per_buffer_, lgpu.logical_id + 1);
            cudaMemcpyAsync(lgpu.bufA, host_init.data(), N_per_buffer_ * sizeof(float), cudaMemcpyHostToDevice, lgpu.stream);
            cudaMemcpyAsync(lgpu.bufB, host_init.data(), N_per_buffer_ * sizeof(float), cudaMemcpyHostToDevice, lgpu.stream);

            logical_gpus_.push_back(lgpu);
        }
    }
}

void LogicalGPUManager::printInfo() const {
    std::cout << "\n=== Logical GPU Mapping ===\n";
    for (const auto& lgpu : logical_gpus_) {
        std::cout << "[Logical GPU " << lgpu.logical_id << "] -> Physical GPU " << lgpu.physical_id
                  << " | Stream: " << lgpu.stream 
                  << " | BufA: " << lgpu.bufA
                  << " | BufB: " << lgpu.bufB
                  << " | BufC: " << lgpu.bufC
                  << std::endl;
    }
    std::cout << "=============================\n";
}
