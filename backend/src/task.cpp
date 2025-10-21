#include "task.hpp"
#include <iostream>

void Task::runAllReduceTask(Mesh& mesh) {
    std::cout << "[Task] Running AllReduce task on mesh with " 
              << mesh.getNumLogicalGPUs() << " logical GPUs." << std::endl;

    mesh.allReduce();

    std::cout << "[Task] AllReduce completed successfully." << std::endl;
}
