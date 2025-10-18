#pragma once
#include <vector>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>

class Mesh {
public:
    Mesh(int num_physical = 1, int logicals_per_phys = 1);

    int getTotalLogical() const { return total_logical_; }
    int getLogicalsPerPhysical() const { return logicals_per_phys_; }
    int getPhysicalCount() const { return num_physical_; }

    int logicalToPhysical(int logical_id) const;

    // Simulated intra-physical AllReduce
    void simulateAllReduce(int logical_id, int size);

    // Inter-physical NCCL AllReduce placeholder
    void interPhysicalAllReduce(int logical_id, int size);

    std::vector<float*>& getBuffers() { return buffers_; }

    ~Mesh();

private:
    int num_physical_;
    int logicals_per_phys_;
    int total_logical_;
    std::vector<cudaStream_t> streams_;
    std::vector<float*> buffers_;
    std::vector<ncclComm_t> nccl_comms_;
};
