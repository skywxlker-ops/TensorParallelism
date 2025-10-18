#include "mesh.hpp"

// Constructor
Mesh::Mesh(int num_physical, int logicals_per_phys)
    : num_physical_(num_physical),
      logicals_per_phys_(logicals_per_phys),
      total_logical_(num_physical * logicals_per_phys) 
{
    std::cout << "[Mesh] Initialized " << total_logical_ << " logical GPUs.\n";

    buffers_.resize(total_logical_);
    streams_.resize(total_logical_);

    for (int i = 0; i < total_logical_; ++i) {
        int phys = logicalToPhysical(i);
        cudaSetDevice(phys);
        cudaStreamCreate(&streams_[i]);
        cudaMalloc(&buffers_[i], sizeof(float) * 5);

        // initialize buffer to zero
        cudaMemset(buffers_[i], 0, sizeof(float) * 5);
    }

    // Initialize NCCL for physical devices
    nccl_comms_.resize(num_physical_);
    ncclCommInitAll(nccl_comms_.data(), num_physical_, nullptr);
}

int Mesh::logicalToPhysical(int logical_id) const {
    return logical_id / logicals_per_phys_;
}

// Simple simulated intra-physical AllReduce
void Mesh::simulateAllReduce(int logical_id, int size) {
    float host_buffer[5] = {0};
    cudaMemcpy(host_buffer, buffers_[logical_id], sizeof(float)*size, cudaMemcpyDeviceToHost);

    // accumulate with all logicals on the same physical device
    int phys = logicalToPhysical(logical_id);
    for (int i = phys*logicals_per_phys_; i < (phys+1)*logicals_per_phys_; ++i) {
        float temp[5];
        cudaMemcpy(temp, buffers_[i], sizeof(float)*size, cudaMemcpyDeviceToHost);
        for(int j=0;j<size;j++) host_buffer[j] += temp[j];
    }

    // copy back result to each logical buffer
    for (int i = phys*logicals_per_phys_; i < (phys+1)*logicals_per_phys_; ++i) {
        cudaMemcpy(buffers_[i], host_buffer, sizeof(float)*size, cudaMemcpyHostToDevice);
    }
    std::cout << "[Mesh] Simulated intra-physical AllReduce done for physical " << phys << ".\n";
}

// Placeholder for inter-physical NCCL AllReduce
void Mesh::interPhysicalAllReduce(int logical_id, int size) {
    std::cout << "[Mesh] Inter-physical NCCL AllReduce placeholder.\n";
}

// Destructor
Mesh::~Mesh() {
    for (auto& buf : buffers_) cudaFree(buf);
    for (auto& s : streams_) cudaStreamDestroy(s);
    for (auto& comm : nccl_comms_) ncclCommDestroy(comm);
}
