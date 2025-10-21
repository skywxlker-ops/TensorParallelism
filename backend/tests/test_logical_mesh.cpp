#include "task.hpp"
#include <iostream>

int main() {
    std::cout << "[Test] Initializing Mesh..." << std::endl;

    // Mode 1: Logical simulation
    Mesh logical_mesh(2, 2, MeshMode::LOGICAL_SIM);
    Task::runAllReduceTask(logical_mesh);

    std::cout << std::endl;

    // Mode 2: Physical GPU path
    Mesh physical_mesh(2, 1, MeshMode::PHYSICAL);
    Task::runAllReduceTask(physical_mesh);

    std::cout << "\n[Test] Completed all modes.\n";
    return 0;
}
