#include "task.hpp"
#include <cuda_runtime.h>
#include <iostream>

void runAllReduceTask(Mesh& mesh, std::vector<float*>& buffers, int N){
    std::cout<<"[Task] Running AllReduce task on mesh with size: "<<mesh.size()<<std::endl;
    mesh.simulateAllReduce(buffers,N);
    std::cout<<"[Task] AllReduce completed. Result:"<<std::endl;

    // print one buffer for verification
    std::vector<float> hostBuf(N);
    cudaMemcpy(hostBuf.data(), buffers[0], sizeof(float)*N, cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++) std::cout<<hostBuf[i]<<" ";
    std::cout<<std::endl;
}
