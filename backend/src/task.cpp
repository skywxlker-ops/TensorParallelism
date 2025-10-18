#include "task.hpp"
#include "mesh.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

void runAllReduceTask(Mesh& mesh) {
    int totalLogical = mesh.getTotalLogical();
    std::cout << "[Task] Running AllReduce task on mesh with size: " 
              << mesh.getSize() << std::endl;

    if (totalLogical <= 0) {
        std::cerr << "[Task] No logical GPUs available!" << std::endl;
        return;
    }

    int bufSize = mesh.getBufferSize();
    std::vector<float> hostBuf(bufSize, 0.0f);

    // Step 1: Gather and sum
    for (int i = 0; i < totalLogical; ++i) {
        std::vector<float> temp(bufSize);
        cudaMemcpy(temp.data(), mesh.getBuffer(i),
                   bufSize * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < bufSize; ++j) {
            hostBuf[j] += temp[j];
        }
    }

    // Step 2: Scatter back
    for (int i = 0; i < totalLogical; ++i) {
        cudaMemcpy(mesh.getBuffer(i), hostBuf.data(),
                   bufSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "[Task] AllReduce completed. Result:\n";
    for (int j = 0; j < bufSize; ++j) {
        std::cout << hostBuf[j] << " ";
    }
    std::cout << std::endl;
}
