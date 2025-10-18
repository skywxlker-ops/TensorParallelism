#include "task.hpp"
#include "mesh.hpp"
#include <iostream>

int main() {
    Mesh mesh(2, 2, 5); // 2 physical GPUs, 2 logical per physical, buffer size 5
    runAllReduceTask(mesh);

    std::cout << "Test completed." << std::endl;
    return 0;
}
