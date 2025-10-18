#include "task.hpp"
#include "mesh.hpp"
#include <iostream>

int main() {
    Mesh mesh(2, 2); // 2 physical GPUs, 2 logical per physical
    runAllReduceTask(mesh);
    std::cout << "Test completed.\n";
    return 0;
}
