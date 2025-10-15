#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "logical_nccl_sim.hpp"

struct LogicalGPU {
    int logical_id;
    int physical_id;
    float* bufA;
    float* bufB;
    float* bufC;
    cudaStream_t stream;

    LogicalGPU() 
        : logical_id(-1), physical_id(-1),
          bufA(nullptr), bufB(nullptr), bufC(nullptr), stream(nullptr) {}
};

class LogicalGPUManager {
public:
    void init(int logical_per_phys, size_t N_per_buffer);
    void printInfo() const;
    std::vector<LogicalGPU>& getGPUs() { return logical_gpus_; }

private:
    std::vector<LogicalGPU> logical_gpus_;
    size_t N_per_buffer_;
};
