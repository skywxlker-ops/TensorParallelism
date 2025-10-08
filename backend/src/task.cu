#include "task.hpp"
#include <iostream>
#include <thread>

__global__ void initTensorKernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        data[i] = val;
}

void Task::initTensors(std::vector<float*>& d_data, Mesh& mesh, int num_elements) {
    int num_gpus = mesh.size();
    d_data.resize(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        set_device(i, true);
        CUDA_CHECK(cudaMalloc(&d_data[i], num_elements * sizeof(float)));

        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        initTensorKernel<<<blocks, threads>>>(d_data[i], float(i + 1), num_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Task::runAllReduce(Mesh& mesh, std::vector<float*>& d_data, int num_elements) {
    int num_gpus = mesh.size();
    std::cout << "[Task] Performing AllReduce across " << num_gpus << " GPUs..." << std::endl;

    std::vector<std::thread> threads(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        threads[i] = std::thread([&, i]() {
            set_device(i, true);
            mesh.allReduce(d_data[i], num_elements);
        });
    }

    for (auto& t : threads) t.join();

    std::vector<float> h_output(num_elements);
    for (int i = 0; i < num_gpus; ++i) {
        set_device(i, true);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_data[i], num_elements * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[GPU " << i << "] Output: ";
        for (int j = 0; j < 10; ++j) std::cout << h_output[j] << " ";
        std::cout << "..." << std::endl;
    }
}
