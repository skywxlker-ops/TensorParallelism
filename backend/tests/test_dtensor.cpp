#include "mesh.hpp"
#include "dtensor.hpp"
#include <vector>
#include <iostream>

int main() {
    // Initialize Mesh
    Mesh mesh;
    mesh.setMeshShape({2});  // 1D mesh for 2 GPUs
    mesh.createSubGroup("tensor", {0,1}); // All GPUs in one subgroup

    // Create DTensor
    std::vector<int64_t> shape = {8, 4}; // 8x4 tensor
    DTensor t(shape, mesh);

    // Set layout: shard along dim0, replicate along dim1
    t.setLayout({"shard", "replicate"});

    std::cout << "[DTensor Test] Printing GPU slices:" << std::endl;
    t.printSlices();

    return 0;
}
