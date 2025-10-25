#pragma once
#include <iostream>
#include <vector>
#include <nccl.h>

class Mesh {
public:
    Mesh(int num_gpus) : num_gpus_(num_gpus) {
        logical_to_physical_.resize(num_gpus_);
        for (int i = 0; i < num_gpus_; ++i) logical_to_physical_[i] = i;

        // Generate NCCL unique ID
        ncclGetUniqueId(&nccl_id_);
    }

    int getDeviceId(int logicalGpu) const {
        return logical_to_physical_[logicalGpu];
    }

    ncclUniqueId getNCCLId() const {
        return nccl_id_;
    }

    int numGPUs() const { return num_gpus_; }

    void printMesh() const {
        std::cout << "[Mesh] num_gpus: " << num_gpus_ << "\n";
        for (int i = 0; i < num_gpus_; ++i)
            std::cout << " GPU " << i << " logical coords: [" << logical_to_physical_[i] << "]\n";
    }

private:
    int num_gpus_;
    std::vector<int> logical_to_physical_;
    ncclUniqueId nccl_id_;
};
