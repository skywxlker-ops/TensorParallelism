#include <iostream>
#include "dtensor.hpp"
#include "mesh.hpp"

int main() {
    Mesh mesh(2);
    mesh.printInfo();

    std::vector<int64_t> shape = {8,4};

    // row-shard, col-replicate
    DTensor dtensor1(&mesh, 8*4);
    dtensor1.setLayout({"shard","replicate"});
    dtensor1.placeData(nullptr);
    std::cout << "[DTensor] Placement: row-shard, col-replicate\n";
    dtensor1.printSlices();
    std::cout << std::endl;

    // row-replicate, col-shard
    DTensor dtensor2(&mesh, 8*4);
    dtensor2.setLayout({"replicate","shard"});
    dtensor2.placeData(nullptr);
    std::cout << "[DTensor] Placement: row-replicate, col-shard\n";
    dtensor2.printSlices();

    return 0;
}
