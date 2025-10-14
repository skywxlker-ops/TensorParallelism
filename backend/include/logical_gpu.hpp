#pragma once
#include <iostream>
#include <vector>

class LogicalGPU {
public:
    int id;           // Logical GPU ID
    int phys_id;      // Physical GPU it maps to
    size_t offset;    // Start offset in GPU memory
    size_t size;      // Size of memory assigned

    LogicalGPU(int logical_id, int physical_id, size_t mem_offset, size_t mem_size)
        : id(logical_id), phys_id(physical_id), offset(mem_offset), size(mem_size) {}

    void info() const {
        std::cout << "[Logical GPU " << id << "] mapped to Physical GPU " << phys_id
                  << " | Offset: " << offset
                  << " | Size: " << size << std::endl;
    }
};

class LogicalGPUManager {
private:
    std::vector<LogicalGPU> logical_gpus_;

public:
    void createLogicalGPUs(int physical_gpus, int logical_count_per_phys, size_t mem_per_gpu) {
        int lgpu_id = 0;
        for(int phys = 0; phys < physical_gpus; ++phys){
            for(int i = 0; i < logical_count_per_phys; ++i){
                size_t offset = i * mem_per_gpu;
                logical_gpus_.emplace_back(lgpu_id++, phys, offset, mem_per_gpu);
            }
        }
    }

    void printLogicalGPUs() const {
        for(const auto &lgpu : logical_gpus_)
            lgpu.info();
    }

    int totalLogicalGPUs() const { return logical_gpus_.size(); }

    const std::vector<LogicalGPU>& getLogicalGPUs() const { return logical_gpus_; }
};
