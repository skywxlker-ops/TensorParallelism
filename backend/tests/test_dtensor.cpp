#include "mesh.hpp"
#include "dtensor.hpp"
#include <vector>

int main() {
    Mesh mesh;
    mesh.setMeshShape({2}); // 2 GPUs
    mesh.createSubGroup("tensor", {0,1});

    std::vector<int64_t> shape = {8,4};
    DTensor t(shape, mesh);

    // Test shard + replicate
    t.setLayout({"shard","replicate"});
    t.printHostTensor();
    std::cout << "[DTensor Test] GPU slices and placements:\n";
    t.printSlices();

    // Test shard + partial
    t.setLayout({"shard","partial"});
    std::cout << "[DTensor Test] With 'partial' placement:\n";
    t.printSlices();

    return 0;
}
