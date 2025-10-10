#include "mesh.hpp"
#include "dtensor.hpp"
#include <vector>

int main() {
    Mesh mesh;
    mesh.setMeshShape({2});
    mesh.createSubGroup("tensor", {0,1});

    std::vector<int64_t> shape = {8,4};
    DTensor t(shape, mesh);
    t.setLayout({"shard","replicate"});

    t.printHostTensor();  // prints only shape
    t.placeData(nullptr);

    std::cout << "[DTensor Test] Printing GPU slices and placements:" << std::endl;
    t.printSlices();

    return 0;
}
