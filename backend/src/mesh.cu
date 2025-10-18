#include "mesh.hpp"

Mesh::Mesh(int num_logical, int logicals_per_phys)
    : num_physical_((num_logical+logicals_per_phys-1)/logicals_per_phys),
      logicals_per_phys_(logicals_per_phys),
      total_logical_(num_logical)
{
    for(int i=0;i<total_logical_;i++){
        logicalToPhysical[i] = i / logicals_per_phys_;
        logical_coords_[i] = {i}; // 1D coordinate
    }
    std::cout << "[Mesh] Initialized " << total_logical_ << " logical GPUs." << std::endl;
}

void Mesh::simulateAllReduce(std::vector<float*>& buffers, int N){
    std::vector<float> hostSum(N,0.f);

    // Copy each GPU buffer to host and sum
    for(auto buf : buffers){
        std::vector<float> tmp(N);
        cudaMemcpy(tmp.data(), buf, sizeof(float)*N, cudaMemcpyDeviceToHost);
        for(int i=0;i<N;i++) hostSum[i] += tmp[i];
    }

    // Copy summed result back to each GPU
    for(auto buf : buffers){
        cudaMemcpy(buf, hostSum.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    }

    std::cout << "[Mesh] Simulated intra-physical AllReduce done." << std::endl;
}
