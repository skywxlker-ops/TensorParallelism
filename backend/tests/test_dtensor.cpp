#include "mesh.hpp"
#include "dtensor.hpp"
#include <vector>

int main() {
    // Initialize mesh
    Mesh mesh;
    mesh.setMeshShape({2}); // 1D mesh for 2 GPUs
    mesh.createSubGroup("tensor", {0,1});

    // Define a tensor of shape 8x4
    std::vector<int64_t> shape = {8,4};
    DTensor t(shape, mesh);

    // Set layout: shard rows, replicate columns (can mark partial for demonstration)
    t.setLayout({"shard","replicate"}); 

    // Print original host tensor shape
    t.printHostTensor();

    // Compute slices and placements
    t.placeData(nullptr);

    // Print slices and placements for each GPU
    std::cout << "[DTensor Test] Printing GPU slices and placements:" << std::endl;
    t.printSlices();

    // Example: demonstrating 'partial' (optional)
    std::vector<int64_t> shape2 = {8,4};
    DTensor t_partial(shape2, mesh);
    t_partial.setLayout({"shard","partial"});
    t_partial.placeData(nullptr);

    std::cout << "[DTensor Test] With 'partial' placement:" << std::endl;
    t_partial.printSlices();

    return 0;
}
