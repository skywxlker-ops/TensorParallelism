#include "task.hpp"
#include <vector>
#include <iostream>

__global__ void initTensor(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

void Task::runAllReduce(Mesh& mesh) {
    int num_gpus = mesh.size();
    int num_elements = 4;
    size_t bytes = num_elements * sizeof(float);

    std::vector<float*> d_data(num_gpus);
    std::vector<float> h_output(num_elements);

    // Allocate & initialize data
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], bytes);
        initTensor<<<1, num_elements>>>(d_data[i], float(i + 1), num_elements);
    }

    std::cout << "[Task] Performing AllReduce across " << num_gpus << " GPUs..." << std::endl;

    // AllReduce: sum across GPUs
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; ++i) {
        NCCL_CHECK(ncclAllReduce(
            d_data[i], d_data[i],
            num_elements, ncclFloat, ncclSum,
            mesh.getComm(i), 0
        ));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Copy result back to host and print
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(h_output.data(), d_data[i], bytes, cudaMemcpyDeviceToHost);
        std::cout << "[GPU " << i << "] Output: ";
        for (float v : h_output) std::cout << v << " ";
        std::cout << std::endl;
    }

    for (int i = 0; i < num_gpus; ++i) cudaFree(d_data[i]);
}
