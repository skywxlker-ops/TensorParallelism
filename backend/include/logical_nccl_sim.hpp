// logical_nccl_sim.hpp
#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace SimNCCL {
    inline void allreduce(std::vector<float*>& buffers, int buffer_size) {
        std::vector<float> host_sum(buffer_size, 0.0f);

        // gather & sum
        std::vector<float> host_buf(buffer_size);
        for(auto buf : buffers) {
            cudaMemcpy(host_buf.data(), buf, buffer_size*sizeof(float), cudaMemcpyDeviceToHost);
            for(int i=0; i<buffer_size; ++i) host_sum[i] += host_buf[i];
        }

        // scatter sum back to all buffers
        for(auto buf : buffers) {
            cudaMemcpy(buf, host_sum.data(), buffer_size*sizeof(float), cudaMemcpyHostToDevice);
        }

        std::cout << "[SimNCCL] Logical GPU completed AllReduce.\n";
    }
}
