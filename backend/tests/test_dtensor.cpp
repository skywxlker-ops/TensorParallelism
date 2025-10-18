#include "../include/mesh.hpp"
#include "../include/dtensor.hpp"
#include <vector>

int main(){
    Mesh mesh(4,1); // 4 logical GPUs, 1 per physical

    std::vector<int64_t> shape = {8,4};
    DTensor t(shape, mesh);

    t.setLayout({"shard","replicate"});
    t.printHostTensor();
    std::cout << "[DTensor Test] GPU slices and placements:\n";
    t.printSlices();

    t.setLayout({"shard","partial"});
    std::cout << "[DTensor Test] With 'partial' placement:\n";
    t.printSlices();

    return 0;
}
