#include "task.hpp"
#include "mesh.hpp"
#include <iostream>

int main() {
    std::cout << "[Test] Initializing logical mesh...\n";
    Mesh mesh(2, 2); // 2 physical GPUs, 2 logical per physical
    Task task(mesh, 16); // 16 elements per tensor
    task.run();
    return 0;
}
