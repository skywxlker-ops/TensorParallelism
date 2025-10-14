#include "logical_gpu.hpp"

void LogicalGPUManager::init(int logical_per_phys, size_t mem_per_logical) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[LogicalGPUManager] No physical GPUs found!" << std::endl;
        return;
    }

    std::cout << "[LogicalGPUManager] Found " << device_count << " physical GPU(s)." << std::endl;

    for (int d = 0; d < device_count; ++d) {
        cudaSetDevice(d);
        size_t total_mem = 0, free_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "[GPU " << d << "] Total memory: " << (total_mem / (1024 * 1024)) 
                  << " MB | Free: " << (free_mem / (1024 * 1024)) << " MB" << std::endl;

        // Base memory allocation for all logical GPUs on this physical GPU
        void* base_ptr = nullptr;
        cudaMalloc(&base_ptr, logical_per_phys * mem_per_logical);

        for (int l = 0; l < logical_per_phys; ++l) {
            LogicalGPU lgpu;
            lgpu.physical_id = d;
            lgpu.logical_id = d * logical_per_phys + l;
            cudaStreamCreate(&lgpu.stream);
            lgpu.mem_size = mem_per_logical;
            lgpu.mem_base = static_cast<char*>(base_ptr) + l * mem_per_logical;

            logical_gpus_.push_back(lgpu);
        }
    }
}

void LogicalGPUManager::printInfo() const {
    std::cout << "\n=== Logical GPU Mapping ===" << std::endl;
    for (const auto& lgpu : logical_gpus_) {
        std::cout << "[Logical GPU " << lgpu.logical_id 
                  << "] → Physical GPU " << lgpu.physical_id
                  << " | Stream: " << lgpu.stream
                  << " | Mem Range: " << lgpu.mem_base
                  << " - " << static_cast<void*>((char*)lgpu.mem_base + lgpu.mem_size)
                  << std::endl;
    }
    std::cout << "=============================\n" << std::endl;
}
