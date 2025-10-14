#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>

struct LogicalGPU {
    int logical_id;
    int physical_id;
    cudaStream_t stream;
    void* mem_base;
    size_t mem_size;

    LogicalGPU() : logical_id(-1), physical_id(-1), stream(nullptr), mem_base(nullptr), mem_size(0) {}
};

class LogicalGPUManager {
public:
    void init(int logical_per_phys, size_t mem_per_logical);
    void printInfo() const;
    const std::vector<LogicalGPU>& getLogicalGPUs() const { return logical_gpus_; }

private:
    std::vector<LogicalGPU> logical_gpus_;
};
