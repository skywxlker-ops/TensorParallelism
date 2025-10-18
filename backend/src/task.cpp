#include "task.hpp"
#include <iostream>

void runAllReduceTask(Mesh& mesh) {
    std::cout << "[Task] Running AllReduce task on mesh with size: "
              << mesh.getTotalLogical() << std::endl;

    int size = 5; // size of each buffer
    for(int i=0;i<mesh.getTotalLogical();i++) {
        mesh.simulateAllReduce(i, size);
    }

    std::cout << "[Task] AllReduce completed. Result:\n";

    // print results
    float host_buffer[5];
    for(int i=0;i<mesh.getTotalLogical();i++) {
        cudaMemcpy(host_buffer, mesh.getBuffers()[i], sizeof(float)*size, cudaMemcpyDeviceToHost);
        for(int j=0;j<size;j++) std::cout << host_buffer[j] << " ";
        std::cout << "\n";
    }
}
