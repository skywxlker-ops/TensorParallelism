#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <cuda_runtime.h>

class Mesh {
public:
    Mesh(int num_logical = 1, int logicals_per_phys = 1);

    int size() const { return total_logical_; }

    // Simulated AllReduce across logical GPUs
    void simulateAllReduce(std::vector<float*>& buffers, int N);

    // Logical coordinates of each GPU
    const std::map<int, std::vector<int>>& meshCoords() const { return logical_coords_; }

private:
    int num_physical_;
    int logicals_per_phys_;
    int total_logical_;
    std::map<int,int> logicalToPhysical; // mapping logical->physical
    std::map<int,std::vector<int>> logical_coords_; // simple 1D coord per GPU
};
