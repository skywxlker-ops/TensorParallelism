#include "task.hpp"
#include <iostream>

void runAllReduceTask(Mesh& mesh) {
    std::cout << "[Task] Running AllReduce task on mesh with size: " << mesh.getSize() << "\n";
    mesh.simulateAllReduce();

    std::cout << "[Task] AllReduce completed. Result:\n";
    int total = mesh.getSize();
    int buffer_size = 5; // must match Mesh default

    for(int i = 0; i < total; i++) {
        for(int j = 0; j < buffer_size; j++)
            std::cout << mesh.getBuffer(i)[j] << " ";
        std::cout << "\n";
    }
}
